from flask import Flask, render_template, request, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from utils.signal_analysis import plot_interpolated_df, f as denoise_f
from sklearn.utils.validation import check_is_fitted
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

df = pd.read_csv("C:/Users/amrut/Downloads/CTU-CHB_Physionet.org/database/final.csv")

with open('random_forest_model_new.pkl', 'rb') as file:
    model = pickle.load(file)

processed_signal_path = None

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    global processed_signal_path
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file uploaded", 400

    # file = request.files['file']
    # file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    # file.save(file_path)

    
    df['FHR_denoised'] = df['FHR'].apply(lambda x: denoise_f(df, x) if x <= 50 or x >= 210 else x)

    processed_signal_path = os.path.join(STATIC_FOLDER, 'denoised_signal.png')
    plot_interpolated_df(df)
    plt.savefig(processed_signal_path)
    plt.close()

    return render_template('result_denoise.html', image_url=url_for('static', filename='denoised_signal.png'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        pH = float(request.form['pH'])
        BDecf = float(request.form['BDecf'])
        pCO2 = float(request.form['pCO2'])
        BE = float(request.form['BE'])
        Apgar1 = float(request.form['Apgar1'])
        Apgar5 = float(request.form['Apgar5'])
        GestWeeks = float(request.form['GestWeeks'])
        Weight = float(request.form['Weight'])
        Age = float(request.form['Age'])
        Gravidity = float(request.form['Gravidity'])
        Parity = float(request.form['Parity'])
        Istage = float(request.form['Istage'])
        IIstage = float(request.form['IIstage'])
        Sex = float(request.form['Sex'])
        Presentation = float(request.form['Presentation'])
        Induced = float(request.form['Induced'])
        NoProgress = float(request.form['NoProgress'])
        CKKP = float(request.form['CKKP'])
        DelivType = float(request.form['DelivType'])
        dbID = float(request.form['dbID'])
        RecType = float(request.form['RecType'])
        PosIIst = float(request.form['PosIIst'])
        Sig2Birth = float(request.form['Sig2Birth'])
        Diabetes =  float(request.form['Diabetes'])
        Hypertension =  float(request.form['Hypertension'])
        Preeclampsia =  float(request.form['Preeclampsia'])
        Liq =  float(request.form['Liq'])
        Pyrexia =  float(request.form['Pyrexia'])
        Meconium =  float(request.form['Meconium'])
        MeanFHR = float(request.form['MeanFHR'])
        MeanUC = float(request.form['MeanUC'])
        MedianFHR = float(request.form['MedianFHR'])
        MedianUC = float(request.form['MedianUC'])
        StdFHR = float(request.form['StdFHR'])
        StdUC = float(request.form['StdUC'])
        RMSFHR = float(request.form['RMSFHR'])
        RMSUC = float(request.form['RMSUC'])
        PeakToRMSFHR = float(request.form['PeakToRMSFHR'])
        PeakToRMSUC = float(request.form['PeakToRMSUC'])
        PeakFHR = float(request.form['PeakFHR'])
        PeakUC = float(request.form['PeakUC'])

        input_data= np.array([pH,BDecf,pCO2,BE, Apgar1,Apgar5,GestWeeks,Weight,Sex,Age,Gravidity, Parity,Diabetes, Hypertension, Preeclampsia,Liq,Pyrexia, Meconium, Presentation, Induced,Istage, NoProgress, CKKP,IIstage,DelivType,dbID,RecType, PosIIst,Sig2Birth,
                        MeanFHR, MeanUC, MedianFHR,MedianUC, StdFHR,StdUC, RMSFHR,RMSUC,PeakToRMSFHR,PeakToRMSUC,PeakFHR,PeakUC]).reshape(1,-1)
        


    except ValueError:
        return "Invalid numeric input for Apgar1, Apgar5, or pH", 400


    prediction = model.predict(input_data)[0]
    prediction_text = "C-section" if prediction == 1 else "Normal"

    return render_template('result_prediction.html', prediction=prediction_text, denoised_signal_url=url_for('static', filename='denoised_signal.png'))



if __name__ == '__main__':
    app.run(debug=True)
