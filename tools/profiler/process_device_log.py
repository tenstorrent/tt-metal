#!/usr/bin/env python3

import os
import sys
import csv

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

import plot_setup


CYCLE_COUNT_TO_MILISECS = 1.2e6
BASE_HEIGHT = 200
PER_CORE_HEIGHT = 90

REARRANGED_TIME_CSV = "device_arranged_timestamps.csv"
DEVICE_STATS_TXT = "device_stats.txt"
DEVICE_PERF_HTML = "device_perf.html"
DEVICE_TIME_CSV = "logs/profile_log_device.csv"

DEVICE_PERF_RESULTS = "device_perf_results.tar"

def coreCompare(coreStr):
    x = int(coreStr.split(",")[0])
    y = int(coreStr.split(",")[1])
    return x + y * 100


def generate_analysis_table(analysisData):
    stats = set()
    for analysis in analysisData.keys():
        for stat in analysisData[analysis].keys():
            stats.add(stat)

    stats = sorted(stats)
    return html.Table(
        # Header
        [html.Tr([html.Th("Type")] + [html.Th(f"{stat} [cycles]") for stat in stats])]
        +
        # Body
        [
            html.Tr(
                [html.Td(f"{analysis}")]
                + [html.Td(f"{analysisData[analysis][stat]:.0f}")\
                   if stat in analysisData[analysis].keys() else html.Td("-") for stat in stats]
            )
            for analysis in analysisData.keys()
        ]
    )


def return_available_timer(riscTimers, timerID):
    if timerID in riscTimers.keys():
        return riscTimers[timerID]
    else:
        return ""


def print_arranged_csv(timerVals, timerIDLabels, pcie_slot, freq_text):
    with open(REARRANGED_TIME_CSV, "w") as timersCSV:
        header = ["core_x", "core_y"]
        timeWriter = csv.writer(timersCSV, delimiter=",")

        timeWriter.writerow(["Clock Frequency [GHz]", freq_text])
        timeWriter.writerow(["PCIe slot",pcie_slot])
        timeWriter.writerow(
            ["core x", "core y"]
            + [f"BRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
            + [f"NCRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
        )
        for core in sorted(timerVals.keys(), key=coreCompare):
            coreSplit = core.split(",")
            core_x = coreSplit[0].strip()
            core_y = coreSplit[1].strip()
            timeWriter.writerow(
                [core_x, core_y]
                + [
                    return_available_timer(timerVals[core]["BRISC"], timerIDLabel[0])
                    for timerIDLabel in timerIDLabels
                ]
                + [
                    return_available_timer(timerVals[core]["NCRISC"], timerIDLabel[0])
                    for timerIDLabel in timerIDLabels
                ]
            )

def analyze_stats(timerStats, timerStatsCores):
    FW_START_VARIANCE_THRESHOLD = 1e3
    if int(timerStats["FW start"]["Max"])  > FW_START_VARIANCE_THRESHOLD:
        print(f"NOTE: Variance on FW starts seems too high at : {timerStats['FW start']['Max']} [cycles]")
        print(f"Please reboot the host to make sure the device is not in a bad reset state")

def print_stats_outfile(timerStats, timerStatsCores):
    original_stdout = sys.stdout
    with open(DEVICE_STATS_TXT, "w") as statsFile:
        sys.stdout = statsFile
        print_stats(timerStats, timerStatsCores)
        sys.stdout = original_stdout


def print_stats(timerStats, timerStatsCores):

    numberWidth = 12
    sampleCores = list(timerStatsCores.keys())
    durationTypes = set()
    for coreDurations in timerStatsCores.values():
        for durationType in coreDurations.keys():
            durationTypes.add(durationType)
    for duration in sorted(durationTypes):
        print()
        print(f"=================== {duration} ===================")
        for stat in sorted(timerStats[duration].keys()):
            print(f"{stat:>12} [cycles] = {timerStats[duration][stat]:>13,.0f}")
        print()
        for core_y in range(-3, 11):
            # Print row number
            if core_y > -1 and core_y < 5:
                print(f"{core_y:>2}|| ", end="")
            elif core_y > 5:
                print(f"{core_y-1:>2}|| ", end="")
            else:
                print(f"{' ':>4} ", end="")

            for core_x in range(-1, 12):
                if core_x > -1:
                    if core_y == -3:
                        print(f"{core_x:>{numberWidth}}", end="")
                    elif core_y == -2:
                        print(f"{'=':=>{numberWidth}}", end="")
                    elif core_y == -1:
                        if core_x in [0, 3, 6, 9]:
                            print(f"{f'DRAM{int(core_x/3)}':>{numberWidth}}", end="")
                        else:
                            print(f"{'---':>{numberWidth}}", end="")
                    elif core_y != 5:
                        core = f"{core_x},{core_y}"
                        if core_y > 5:
                            core = f"{core_x},{core_y-1}"
                        if (
                            core in timerStatsCores.keys()
                            and duration in timerStatsCores[core].keys()
                        ):
                            print(
                                f"{timerStatsCores[core][duration]:>{numberWidth},}",
                                end="",
                            )
                        else:
                            print(f"{'X':>{numberWidth}}", end="")
                    else:
                        if core_x in [0, 3, 6, 9]:
                            print(
                                f"{f'DRAM{4 + int(core_x/3)}':>{numberWidth}}", end=""
                            )
                        else:
                            print(f"{'---':>{numberWidth}}", end="")

                else:
                    if core_y == 1:
                        print("ARC", end="")
                    elif core_y == 3:
                        print("PCI", end="")
                    elif core_y > -1:
                        print("---", end="")
                    else:
                        print("   ", end="")

            print()
        print()
        print()
        print()
    for duration in timerStats.keys():
        if duration not in durationTypes:
            print(f"=================== {duration} ===================")
            for stat in sorted(timerStats[duration].keys()):
                print(f"{stat:>12} [cycles] = {timerStats[duration][stat]:>13,.0f}")


def print_help():
    print(
        "Please choose a plot setup class that matches your test kernel profile data."
    )
    print("e.g. : process_device_log.py test_add_two_ints")
    print("Or run default by providing no args")
    print("e.g. : process_device_log.py")


def main(args):
    if len(args) == 1:
        try:
            setup = getattr(plot_setup, args[0])()
            try:
                setup.timerAnalysis.update(setup.timerAnalysisBase)
            except Exception:
                setup.timerAnalysis = setup.timerAnalysisBase
        except Exception:
            print_help()
            return
    elif len(args) == 0:
        try:
            setup = getattr(plot_setup, "test_base")()
            setup.timerAnalysis = setup.timerAnalysisBase
        except Exception:
            print_help()
            return
    else:
        print_help()
        return

    timerVals = {}
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    freq_text = ""
    with open(DEVICE_TIME_CSV, "r") as csvfile:
        freq_text = csvfile.readline().split('at ')[-1].split(' ')[0].strip()
        fields = csvfile.readline()
        for line in csvfile.readlines():
            vals = line.split(",")
            if len(vals) == 6:
                pcie_slot = vals[0].strip()
                col = vals[1].strip()
                row = vals[2].strip()
                risc = vals[3].strip()
                timerID = vals[4].strip()
                cycleCount = int(vals[5].strip())

                core = f"{col},{row}"
                if core in timerVals.keys():
                    if risc in timerVals[core].keys():
                        timerVals[core][risc][timerID] = cycleCount
                    else:
                        timerVals[core][risc] = {timerID: cycleCount}
                else:
                    timerVals[core] = {risc: {timerID: cycleCount}}

    print_arranged_csv(timerVals, setup.timerIDLabels, pcie_slot, freq_text)

    if timerVals:
        maxTime = 0
        minTime = (1 << 64) - 1
        for core in timerVals.keys():
            for risc in timerVals[core].keys():
                for timerID, cycleCount in timerVals[core][risc].items():
                    if cycleCount > maxTime:
                        maxTime = cycleCount
                    if cycleCount < minTime:
                        minTime = cycleCount
        timerStats = {"RunTime" : {"Total" : maxTime - minTime}}
        timerStatsCores = {}
        setupMaxStrLen = 0
        for timerAnalysisSetup in setup.timerAnalysis.keys():
            if len(timerAnalysisSetup) > setupMaxStrLen:
                setupMaxStrLen = len(timerAnalysisSetup)

        for timerAnalysisSetup in setup.timerAnalysis.keys():
            analysisTimeMax = 0
            analysisTimeMin = (1 << 64) - 1
            analysisTimeSum = 0

            countAnalysisTime = 0
            for core in timerVals.keys():
                analysisTime = -1
                setupType = setup.timerAnalysis[timerAnalysisSetup]["type"]
                if setupType == "single":
                    risc = setup.timerAnalysis[timerAnalysisSetup]["risc"]
                    timerID = setup.timerAnalysis[timerAnalysisSetup]["timerID"]
                    if timerID in timerVals[core][risc].keys():
                        analysisTime = timerVals[core][risc][timerID] - minTime
                elif setupType == "diff":
                    startRisc = setup.timerAnalysis[timerAnalysisSetup]["start"]["risc"]
                    startTimerID = setup.timerAnalysis[timerAnalysisSetup]["start"][
                        "timerID"
                    ]
                    endRisc = setup.timerAnalysis[timerAnalysisSetup]["end"]["risc"]
                    endTimerID = setup.timerAnalysis[timerAnalysisSetup]["end"][
                        "timerID"
                    ]
                    if (
                        startRisc in timerVals[core].keys()
                        and endRisc in timerVals[core].keys()
                        and startTimerID in timerVals[core][startRisc].keys()
                        and endTimerID in timerVals[core][endRisc].keys()
                    ):
                        analysisTime = (
                            timerVals[core][endRisc][endTimerID]
                            - timerVals[core][startRisc][startTimerID]
                        )

                if analysisTime != -1:
                    if core in timerStatsCores.keys():
                        timerStatsCores[core][timerAnalysisSetup] = analysisTime
                    else:
                        timerStatsCores[core] = {timerAnalysisSetup: analysisTime}
                    countAnalysisTime += 1
                    analysisTimeSum += analysisTime
                    if analysisTime > analysisTimeMax:
                        analysisTimeMax = analysisTime
                    if analysisTime < analysisTimeMin:
                        analysisTimeMin = analysisTime

            analysisTimeAverage = 0
            if countAnalysisTime:
                analysisTimeAverage = analysisTimeSum / countAnalysisTime

            timerStats[timerAnalysisSetup] = {
                "Average": analysisTimeAverage,
                "Min": analysisTimeMin,
                "Max": analysisTimeMax,
            }

        print_stats(timerStats, timerStatsCores)
        print_stats_outfile(timerStats, timerStatsCores)
        analyze_stats(timerStats, timerStatsCores)

        yVals = sorted(timerVals.keys(), key=coreCompare, reverse=True)
        xVals = {}

        for yVal in yVals:
            for risc in setup.riscTimerCombo.keys():
                if risc in timerVals[yVal].keys():
                    for combo in setup.riscTimerCombo[risc]:
                        xVal = 0
                        if (
                            combo[0] == "START"
                            and combo[1] in timerVals[yVal][risc].keys()
                        ):
                            xVal = timerVals[yVal][risc][combo[1]] - (minTime - 5)
                        elif (
                            combo[1] == "END"
                            and combo[0] in timerVals[yVal][risc].keys()
                        ):
                            xVal = maxTime - timerVals[yVal][risc][combo[0]]
                        elif (
                            combo[0] in timerVals[yVal][risc].keys()
                            and combo[1] in timerVals[yVal][risc].keys()
                        ):
                            xVal = (
                                timerVals[yVal][risc][combo[1]]
                                - timerVals[yVal][risc][combo[0]]
                            )

                            # TODO: Add drop down for time / cycles display
                            # xVal = xVal / CYCLE_COUNT_TO_MILISECS

                        if risc in xVals.keys():
                            if combo in xVals[risc].keys():
                                xVals[risc][combo].append(xVal)
                            else:
                                xVals[risc][combo] = [xVal]
                        else:
                            xVals[risc] = {combo: [xVal]}

        # TODO: Add drop down for time / cycles display
        # layout = go.Layout(xaxis=dict(title="Time [ms]"), yaxis=dict(title="Cores"))

        layout = go.Layout(xaxis=dict(title="Cycle count"), yaxis=dict(title="Cores"))
        fig = go.Figure(layout=layout)

        fig.add_trace(
            go.Bar(
                y=[yVals, [" "] * len(yVals)],
                x=[0] * len(yVals),
                orientation="h",
                showlegend=False,
                marker=dict(color="rgba(255, 255, 255, 0.0)"),
            )
        )
        for risc in setup.riscTimerCombo.keys():
            if risc in xVals.keys():
                for combo in setup.riscTimerCombo[risc]:
                    color = setup.colors[combo[2]]

                    showlegend = True
                    if combo[2] == "blank":
                        showlegend = False

                    fig.add_trace(
                        go.Bar(
                            y=[yVals, [risc] * len(yVals)],
                            x=xVals[risc][combo],
                            orientation="h",
                            name=combo[3],
                            showlegend=showlegend,
                            marker=dict(color=color),
                        )
                    )
        fig.add_trace(
            go.Bar(
                y=[yVals, [""] * len(yVals)],
                x=[0] * len(yVals),
                orientation="h",
                showlegend=False,
                marker=dict(color="rgba(255, 255, 255, 0.0)"),
            )
        )

        fig.update_layout(
            barmode="stack", height=BASE_HEIGHT + PER_CORE_HEIGHT * len(yVals)
        )

        fig.write_html(DEVICE_PERF_HTML)

        os.system(
            f"tar -czf {DEVICE_PERF_RESULTS} {DEVICE_TIME_CSV} {DEVICE_STATS_TXT} {DEVICE_PERF_HTML} {REARRANGED_TIME_CSV}"
        )

        app.layout = html.Div(
            [
                html.H1("Device Performance"),
                html.Br(),
                html.H3("Stats Table"),
                generate_analysis_table(timerStats),
                dcc.Graph(figure=fig),
            ]
        )

        app.run_server(host="0.0.0.0", debug=True)
    else:
        print(f"Empty {DEVICE_TIME_CSV}")


if __name__ == "__main__":
    main(sys.argv[1:])
