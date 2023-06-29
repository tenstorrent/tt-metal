#!/usr/bin/env python3

import os
import csv
from pathlib import Path
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import click
from loguru import logger
from dash import Dash, dcc, html, Input, Output

from process_device_log import import_log_run_stats, generate_plots
from process_host_log import import_host_log_run_stats
import plot_setup

OPS_LOGS_DIR = os.path.abspath("logs/ops")
DEVICE_SIDE_LOG = "profile_log_device.csv"
HOST_SIDE_LOG = "profile_log_host.csv"
OUT_FOLDER = "output/ops"
OUT_NAME = "profile_log_ops"

OPS_CSV_HEADER = [
    "NAME",
    "IS OP",
    "CALL COUNT",
    "GLOBAL CALL COUNT",
    "HOST START TS",
    "HOST END TS",
    "HOST DURATION [ns]",
    "DEVICE START CYCLE",
    "DEVICE END CYCLE",
    "DEVICE DURATION [ns]",
    "CORE COUNT",
    "CALL DEPTH",
    "INPUTS",
    "OUTPUTS",
    "MATH FIDEL.",
    "PARAL. STRAT.",
    "META DATA",
]

HOST_SIDE_STATS = ["Count", "Average"]
HOST_FUNCSTION_HEADER_FORMAT = "{} {}"

ttMetalFunctionsSet = set()


def append_detail_host_time_data(opCandidatePath, call_count, timeDataDict):
    hostLogPath = os.path.join(opCandidatePath, f"{call_count}", HOST_SIDE_LOG)
    if os.path.isfile(hostLogPath):
        hostData = import_host_log_run_stats(hostLogPath)
        for functionName, calls in hostData.items():
            ttMetalFunctionsSet.add(functionName)
            for stat in HOST_SIDE_STATS:
                assert stat in calls["stats"].keys()
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                timeDataDict[functionKey] = int(calls["stats"][stat])


def append_device_time_data(opCandidatePath, call_count, timeDataDict):
    deviceLogPath = os.path.join(opCandidatePath, f"{call_count}", DEVICE_SIDE_LOG)
    if os.path.isfile(deviceLogPath):
        setup = plot_setup.default_setup()
        setup.deviceInputLog = deviceLogPath
        setup.timerAnalysis = {}

        devicesData = import_log_run_stats(setup)
        deviceID = list(devicesData["devices"].keys())[0]  # Assume there is only one device

        timeseriesData = devicesData["devices"][deviceID]["cores"]["DEVICE"]["riscs"]["TENSIX"]["timeseries"]
        start_ID, start_ts, start_risc, start_core = timeseriesData[0]
        end_ID, end_ts, end_risc, end_core = timeseriesData[-1]

        cores = list(devicesData["devices"][deviceID]["cores"].keys())
        cores.remove("DEVICE")

        delta_time = end_ts - start_ts
        delta_time_ns = delta_time / setup.coreFreq

        timeDataDict["DEVICE START CYCLE"] = start_ts
        timeDataDict["DEVICE END CYCLE"] = end_ts
        timeDataDict["DEVICE DURATION [ns]"] = int(delta_time_ns)
        timeDataDict["CORE COUNT"] = len(cores)
    else:
        timeDataDict["DEVICE START CYCLE"] = "-"
        timeDataDict["DEVICE END CYCLE"] = "-"
        timeDataDict["DEVICE DURATION [ns]"] = "-"
        timeDataDict["CORE COUNT"] = "-"


minTime = 0
maxDiff = 0
maxStackSize = 0

op_to_folder = {}
op_flavour_to_count = {}


def parse_ops_logs(opsFolder):
    global minTime, maxDiff, maxStackSize
    ops = {}

    assert os.path.isdir(opsFolder), f"{opsFolder} does no exists. Use -i option to choose the correct logs dir"
    paths = sorted(Path(opsFolder).iterdir(), key=os.path.getmtime, reverse=True)
    assert paths, f"{opsFolder} is empty. Use -i option to choose the correct logs dir"

    for opCandidate in paths:
        opCandidatePath = os.path.join(opsFolder, opCandidate)
        if os.path.isdir(opCandidatePath):
            if "unknown" in str(opCandidate).lower():
                continue
            opLogPath = os.path.join(opCandidatePath, HOST_SIDE_LOG)
            if not os.path.isfile(opLogPath):
                logger.warning(f"Skipped: {opLogPath} dir, no host side log found.")
                continue
            with open(opLogPath, "r") as csvFile:
                csvReader = csv.reader(csvFile, delimiter=",")
                for lineCount, row in enumerate(csvReader):
                    if lineCount > 0:
                        op_folder_name = row[1].strip()
                        op_name = op_folder_name
                        is_op = "No"
                        extractName = re.findall(r".*tt.*tt_metal:*\d*(.*)E*", op_name)
                        if extractName:
                            op_name = extractName.pop()
                            is_op = "Yes"

                        start_ts = int(row[2].strip())
                        end_ts = int(row[3].strip())
                        delta_time = int(row[4].strip())

                        global_call_count = int(row[5].strip())
                        call_count = int(row[6].strip())
                        stack_size = int(row[7].strip())

                        inputs = row[8].strip()
                        outputs = row[9].strip()
                        mathFidelity = row[10].strip()
                        parallelizationStrategy = row[11].strip()
                        preferredName = row[12].strip().split("tt::tt_metal::")[-1]
                        metadata = row[13].strip()

                        if preferredName:
                            if is_op == "Yes":
                                op_name = preferredName
                            else:
                                op_name += "_" + preferredName

                        op_to_folder[op_name] = op_folder_name
                        if op_name in op_flavour_to_count.keys():
                            op_flavour_to_count[op_name] += 1
                        else:
                            op_flavour_to_count[op_name] = 1

                        if minTime == 0:
                            minTime = start_ts
                        elif minTime > start_ts:
                            minTime = start_ts

                        if maxDiff < delta_time:
                            maxDiff = delta_time

                        if stack_size > maxStackSize:
                            maxStackSize = stack_size

                        timeDataDict = {
                            "CALL COUNT": op_flavour_to_count[op_name],
                            "_OP CALL COUNT": call_count,
                            "IS OP": is_op,
                            "GLOBAL CALL COUNT": global_call_count,
                            "HOST START TS": start_ts,
                            "HOST END TS": end_ts,
                            "CALL DEPTH": stack_size,
                            "INPUTS": inputs,
                            "OUTPUTS": outputs,
                            "MATH FIDEL.": mathFidelity,
                            "PARAL. STRAT.": parallelizationStrategy,
                            "HOST DURATION [ns]": delta_time,
                            "META DATA": metadata,
                        }

                        append_device_time_data(opCandidatePath, call_count, timeDataDict)
                        append_detail_host_time_data(opCandidatePath, call_count, timeDataDict)

                        if op_name in ops.keys():
                            ops[op_name].append(timeDataDict)
                        else:
                            ops[op_name] = [timeDataDict]
                    else:
                        assert "Start timer count [ns]" in row[2], f"CSV {opLogPath} has bad header format"
    return ops


preFig = go.Figure()


def run_dashbaord_webapp(ops, opsFolder, port=None):
    global preFig
    curveDict = {}
    curveNumber = 0
    fig = go.Figure()
    for op, opCalls in ops.items():
        xVals = []
        yVals = []
        Xs = []
        Ys = []
        Cs = []
        Ss = []
        diffs = []
        names = []
        for opCall in opCalls:
            s = opCall["HOST START TS"] - minTime
            e = opCall["HOST END TS"] - minTime
            c = opCall["_OP CALL COUNT"]
            callDepth = opCall["CALL DEPTH"]
            y = 1 + (0.2 / maxStackSize) * (maxStackSize - callDepth + 1)
            diff = opCall["HOST DURATION [ns]"]
            ps = opCall["META DATA"]
            m = (s + e) // 2
            xVals += [None, s, e, e, s, s]
            yVals += [None, 0, 0, y, y, 0]
            Xs += [m]
            Ys += [y]
            Cs += [c]
            diffs += [diff / 1e9]
            names += [op]
            Ss += [ps]

            curveDict[curveNumber] = {"op": op, "callCount": c}
            curveNumber += 1

        fig.add_trace(go.Scatter(x=xVals, y=yVals, name=op, hoverinfo="none", mode="none", fill="toself"))

        fig.add_trace(
            go.Scatter(
                x=Xs,
                y=Ys,
                name="",
                customdata=np.stack((names, Cs, diffs, Ss), axis=-1),
                hovertemplate="<br>".join(
                    [
                        "Op: %{customdata[0]}",
                        "Call: %{customdata[1]}",
                        "Duration: %{customdata[2]:.3f} s",
                        "Meta: %{customdata[3]}",
                    ]
                ),
                mode="markers",
                marker_size=5,
                hoverlabel=dict(
                    bgcolor="white",
                ),
                marker_color="black",
                hoverinfo="x",
                showlegend=True,
                opacity=0.5,
            )
        )
        fig.update_layout(
            xaxis=dict(
                range=[-1e7, maxDiff + 1e7],
                rangeslider=dict(
                    visible=True,
                ),
            ),
            yaxis=dict(
                visible=False,
            ),
        )

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div(
        [
            html.H5(f"OPs:", id="text"),
            dcc.Graph(figure=fig, id="plot"),
            dcc.Graph(figure=go.Figure(), id="plot-2"),
        ]
    )

    @app.callback(Output("plot-2", "figure"), [Input("plot", "hoverData")])
    def plot_device_data(hoverData):
        global preFig
        fig = preFig
        if hoverData and "points" in hoverData.keys():
            if len(hoverData["points"]) > 0:
                if "customdata" in hoverData["points"][0].keys():
                    op = hoverData["points"][0]["customdata"][0]
                    opFolder = op_to_folder[op]
                    callCount = hoverData["points"][0]["customdata"][1]
                    filePath = f"{opsFolder}/{opFolder}/{callCount}/{DEVICE_SIDE_LOG}"
                    if os.path.isfile(filePath):
                        setup = plot_setup.default_setup()
                        setup.deviceInputLog = filePath
                        setup.timerAnalysis = {}

                        devicesData = import_log_run_stats(setup)
                        figs = generate_plots(devicesData, setup, saveFigure=False)
                        for fig in figs.values():
                            preFig = fig

        return fig

    if port:
        app.run_server(host="0.0.0.0", port=port, debug=True)
    else:
        app.run_server(host="0.0.0.0", debug=True)


def print_ops_csv(ops, opsFolder, outputFolder, date, nameAppend):
    outFolder = OUT_FOLDER
    if outputFolder:
        outFolder = outputFolder

    outFolder = os.path.abspath(outFolder)

    os.system(f"rm -rf {outFolder}; mkdir -p {outFolder}")

    if date:
        name = f"{datetime.now().strftime('%Y_%d_%m_%H_%M')}"
    else:
        name = OUT_NAME

    if nameAppend:
        name += f"_{nameAppend}"

    opsCSVPath = f"{outFolder}/{name}.csv"

    os.system(f"cd {opsFolder} && tar -czf ../{name}.tgz . && mv ../{name}.tgz {outFolder}")

    with open(opsCSVPath, "w") as opsCSV:
        opsWriter = csv.writer(opsCSV, delimiter=",")
        hostFunctions = []
        for functionName in sorted(ttMetalFunctionsSet):
            for stat in HOST_SIDE_STATS:
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                if "Count" not in functionKey:
                    hostFunctions.append(f"{functionKey} [ns]")
                else:
                    hostFunctions.append(functionKey)
        opsWriter.writerow(OPS_CSV_HEADER + hostFunctions)

        for op, opCalls in ops.items():
            for opCall in opCalls:
                opsROW = [op]
                for item in OPS_CSV_HEADER:
                    if item != "NAME":
                        assert item in opCall.keys(), item
                        opsROW.append(opCall[item])
                for functionName in sorted(ttMetalFunctionsSet):
                    for stat in HOST_SIDE_STATS:
                        functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                        if functionKey in opCall.keys():
                            opsROW.append(opCall[functionKey])
                        else:
                            opsROW.append("0")
                opsWriter.writerow(opsROW)


@click.command()
@click.option(
    "-i",
    "--ops-folder",
    type=click.Path(exists=True, dir_okay=True),
    help="Ops profiler logs folder",
)
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("-p", "--port", type=int, help="Dashboard webapp port")
@click.option("--webapp", default=False, is_flag=True, help="Run dashboard webapp")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
def main(ops_folder, output_folder, name_append, port, webapp, date):
    opsFolder = OPS_LOGS_DIR
    if ops_folder:
        opsFolder = os.path.abspath(ops_folder)

    ops = parse_ops_logs(opsFolder)
    print_ops_csv(ops, opsFolder, output_folder, date, name_append)

    if webapp:
        # TODO: Works but needs more refining
        logger.info("Web app dashboard is a work in progress. Don't be alarmed by bugs and glitches!")
        run_dashbaord_webapp(ops, opsFolder, port)


if __name__ == "__main__":
    main()
