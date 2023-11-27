import streamlit as st
import pickle
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
Gears=st.sidebar.selectbox("What is the gear type of your car:",(5,6,7,8))
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
Displacement_cc=st.sidebar.slider("What is the displacement of your car", 900,3000, step=100)
typen=st.sidebar.selectbox("Select the type of the car", ('Used', "Employee's car", 'New', 'Demonstration', 'Pre-registered'))



ds13_model=joblib.load('FINAL_ELSTIC_NET_MLD_PROJECT.pkl')


my_dict = {
        "make_model": car_model,
         "hp_kW": hp,
         "Gears": Gears,
         "km": km,
         "age": age,
         "Displacement_cc": Displacement_cc,
         "Type": typen
 
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)

#df2 = ds13_transformer.transform(df)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = ds13_model.predict(df)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))

# Dataframe
df=pd.read_csv("final_scout_not_dummy.csv")