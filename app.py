from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
odi_model = pickle.load(open('odi_model.pkl', 'rb'))
t20i_model = pickle.load(open('t20i_model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    # return("Hello World")


@app.route('/odi', methods=['GET', 'POST'])
def render_odi():
    return render_template('odi_index.html')

@app.route('/t20i', methods=['GET', 'POST'])
def render_t20i():
    return render_template('t20i_index.html')


@app.route('/predict', methods=['POST'])
def predict_score():

    bat_team = request.form.get('batting_team')

    bowl_team = request.form.get('bowling_team')

    venue = request.form.get('venue')

    overs = float(request.form.get('overs'))
    runs = int(request.form.get('runs'))
    wickets = int(request.form.get('wickets'))
    runs_last_5 = int(request.form.get('runs_last_5'))
    wickets_last_5 = int(request.form.get('wickets_last_5'))

    input_df = pd.DataFrame({'venue':[venue],'bat_team':[bat_team],'bowl_team':[bowl_team],'runs':[runs],'wickets':[wickets],'overs':[overs],'runs_last_5':[runs_last_5],'wickets_last_5':[wickets_last_5]})

    result = int(model.predict(input_df)[0])

    # return str(result)
    return render_template('index.html', result=result, team = bat_team)


@app.route('/odi-predict', methods=['POST'])
def odi_predict_score():

    odi_bat_team = request.form.get('batting_team')

    odi_bowl_team = request.form.get('bowling_team')

    odi_venue = request.form.get('venue')

    odi_overs = float(request.form.get('overs'))
    odi_runs = int(request.form.get('runs'))
    odi_wickets = int(request.form.get('wickets'))
    odi_runs_last_5 = int(request.form.get('runs_last_5'))
    odi_wickets_last_5 = int(request.form.get('wickets_last_5'))

    odi_input_df = pd.DataFrame({'venue':[odi_venue],'bat_team':[odi_bat_team],'bowl_team':[odi_bowl_team],'runs':[odi_runs],'wickets':[odi_wickets],'overs':[odi_overs],'runs_last_5':[odi_runs_last_5],'wickets_last_5':[odi_wickets_last_5]})

    odi_result = int(odi_model.predict(odi_input_df)[0])

    # return str(result)
    return render_template('odi_index.html', result=odi_result, team = odi_bat_team)


@app.route('/t20i-predict', methods=['POST'])
def t20i_predict_score():

    t20i_bat_team = request.form.get('batting_team')

    t20i_bowl_team = request.form.get('bowling_team')

    t20i_venue = request.form.get('venue')

    t20i_overs = float(request.form.get('overs'))
    t20i_runs = int(request.form.get('runs'))
    t20i_wickets = int(request.form.get('wickets'))
    t20i_runs_last_5 = int(request.form.get('runs_last_5'))
    t20i_wickets_last_5 = int(request.form.get('wickets_last_5'))

    t20i_input_df = pd.DataFrame({'venue':[t20i_venue],'bat_team':[t20i_bat_team],'bowl_team':[t20i_bowl_team],'runs':[t20i_runs],'wickets':[t20i_wickets],'overs':[t20i_overs],'runs_last_5':[t20i_runs_last_5],'wickets_last_5':[t20i_wickets_last_5]})

    t20i_result = int(t20i_model.predict(t20i_input_df)[0])

    # return str(result)
    return render_template('t20i_index.html', result=t20i_result, team = t20i_bat_team)


if __name__ == '__main__':
    app.run(debug=True)
    
