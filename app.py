import fastapi
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from pycaret.regression import load_model, predict_model, setup
from pydantic import BaseModel
import pandas as pd

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('mortality_prediction_pipeline')
all_data=pd.read_csv("Mortality_09_UP.data",nrows=25175)
columns = ["age","sex", "highest_qualification", "rural", "disability_status", "is_water_filter", "chew", "smoke", "alcohol","treatment_source"]
columns_mortality_factors = ["is_water_filter", "chew", "smoke", "alcohol","treatment_source"]
death = all_data[columns].copy()
for column in columns:
    death[column].fillna(death[column].mode()[0], inplace=True)

s = setup(death, target='age',session_id=110)
# class PredData(BaseModel):
#     sex : float
#     highest_qualification : float
#     rural : float
#     disability_status : float
#     is_water_filter : float
#     chew : float
#     smoke : float
#     alcohol : float
#     treatment_source : float

@app.post("/predict/")
async def predict_age(data : dict):
    data_with_factors = {k : [data[k],] for k in data}
    data_without_factors = {}
    for k in data.keys():
        if(k in columns_mortality_factors):
            data_without_factors[k] = [0.0,]
        else:
            data_without_factors[k] = [data[k],]
    x = pd.DataFrame.from_dict(data_with_factors)
    y = pd.DataFrame.from_dict(data_without_factors)
    preds_wf = predict_model(model, data=x)["prediction_label"]
    preds_nf = predict_model(model, data=y)["prediction_label"]
    
    return {"age" : preds_wf, "no_factors_age" : preds_nf, "difference" : preds_wf - preds_nf}
