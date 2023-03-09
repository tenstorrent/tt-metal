#!/usr/bin/env python3

import os
import sys

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

import plot_setup


CYCLE_COUNT_TO_MILISECS = 1.2e6
BASE_HEIGHT = 200
PER_CORE_HEIGHT = 90


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
        [html.Tr([html.Th("Type")] + [html.Th(stat) for stat in stats])]
        +
        # Body
        [
            html.Tr(
                [html.Td(analysis)]
                + [html.Td(analysisData[analysis][stat]) for stat in stats]
            )
            for analysis in analysisData.keys()
        ]
    )


def print_stats_outfile(timerStats, timerStatsCores):
    original_stdout = sys.stdout
    with open("device_stats.txt", "w") as statsFile:
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
    for duration in durationTypes:
        print()
        print(f"=================== {duration} ===================")
        for stat in timerStats[duration].keys():
            print(f"{stat:>20} = {timerStats[duration][stat]}")
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
                        if core in timerStatsCores.keys() and duration in timerStatsCores[core].keys():
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


def print_help():
    print(
        "Please choose a plot setup class that matches your test kernel profile data."
    )
    print("e.g. : psotproc_kernel_log.py test_add_two_ints")
    print("Or run default by providing no args")
    print("e.g. : psotproc_kernel_log.py")


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

    with open("./profile_log_kernel.csv", "r") as csvfile:
        freq = csvfile.readline()
        fields = csvfile.readline()
        for line in csvfile.readlines():
            vals = line.split(",")
            if len(vals) == 6:
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

        timerStats = {}
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
                "Average [cycles]": f"{analysisTimeAverage:<14,.2f}",
                "Min [cycles]": f"{analysisTimeMin:<11,}",
                "Max [cycles]": f"{analysisTimeMax:<11,}",
            }

        print_stats(timerStats, timerStatsCores)
        print_stats_outfile(timerStats, timerStatsCores)

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
                            marker=dict(
                                color=color,
                            ),
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

        fig.write_html("kernel_perf.html")

        app.layout = html.Div(
            [
                html.H1("Kernel Profiler Plot"),
                html.Br(),
                html.H3("Stats Table"),
                generate_analysis_table(timerStats),
                dcc.Graph(figure=fig),
            ]
        )

        app.run_server(host="0.0.0.0", debug=True)
    else:
        print("Empty profile_log_kernel.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
