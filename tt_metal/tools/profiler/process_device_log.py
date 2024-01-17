#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import inspect
import csv
import json
from datetime import datetime

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import seaborn as sns
import click

from tt_metal.tools.profiler.common import PROFILER_ARTIFACTS_DIR
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
import tt_metal.tools.profiler.dummy_refresh as dummy_refresh


# TODO(MO): Grab this from the core_descriptor yaml files
NON_COMPUTE_ROW = 9


def coreCompare(core):
    if type(core) == str:
        return (1 << 64) - 1
    x = core[0]
    y = core[1]
    return x + y * 100


def generate_analysis_table(analysisData, setup):
    stats = setup.displayStats
    return html.Div(
        [
            html.H6("Duration Stats Table"),
            html.P("Duration is the period between two recorded events"),
            html.P(
                "T0 is the the 0 refrence point, corresponding to the earliest core that reports a BRISC FW start marker"
            ),
            html.Table(
                # Header
                [
                    html.Tr(
                        [html.Th("Duration Name")]
                        +
                        # TODO(MO): Issue 799
                        # [html.Th("Analysis Type")] +\
                        # [html.Th("Across")] +\
                        [html.Th(f"{stat} [cycles]") if stat not in ["Count"] else html.Th(f"{stat}") for stat in stats]
                    )
                ]
                +
                # Body
                [
                    html.Tr(
                        [html.Td(f"{analysis}")]
                        +
                        # TODO(MO): Issue 799
                        # [html.Td(f"{setup.timerAnalysis[analysis]['type']}")] +\
                        # [html.Td(f"{setup.timerAnalysis[analysis]['across']}")] +\
                        [
                            html.Td(f"{analysisData[analysis]['stats'][stat]:,.0f}")
                            if stat in analysisData[analysis]["stats"].keys()
                            else html.Td("-")
                            for stat in stats
                        ]
                    )
                    for analysis in analysisData.keys()
                ]
            ),
        ]
    )


# Note if multiple instances are present, all are returned space delimited
# Further analysis has to be done on the excel side
def return_available_timer(risc, coreTimeseries, timerIDLabels):
    resList = []
    for desiredTimerID, label in timerIDLabels:
        res = ""
        if risc in coreTimeseries.keys():
            timeseries = coreTimeseries[risc]["timeseries"]
            for timerID, timestamp, *metaData in timeseries:
                if timerID == desiredTimerID:
                    if res:
                        res = f"{res} {timestamp}"
                    else:
                        res = f"{timestamp}"
        resList.append(res)
    return resList


class TupleEncoder(json.JSONEncoder):
    def _preprocess_tuple(self, obj):
        if isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, device_post_proc_config.default_setup):
            objDict = {}
            for attr in dir(obj):
                if "__" not in attr:
                    objDict[attr] = getattr(obj, attr)
            return objDict
        elif isinstance(obj, dict):
            return {self._preprocess_tuple(k): self._preprocess_tuple(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._preprocess_tuple(i) for i in obj]
        return obj

    def default(self, obj):
        if isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, device_post_proc_config.default_setup):
            objDict = {}
            for attr in dir(obj):
                if "__" not in attr:
                    objDict[attr] = getattr(obj, attr)
            return objDict
        return super().default(obj)

    def iterencode(self, obj):
        return super().iterencode(self._preprocess_tuple(obj))


def print_chrome_tracing_json(devicesData, setup):
    chromeTraces = []
    riscToNum = {"BRISC": 1, "NCRISC": 2, "TRISC_0": 3, "TRISC_1": 4, "TRISC_2": 5}
    for device in devicesData["devices"].keys():
        minTime = devicesData["devices"][device]["metadata"]["global_min"]["ts"]
        for timerID, timestamp, risc, core in devicesData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"][
            "timeseries"
        ]:
            x, y = core
            traceID = riscToNum[risc] + x * 100 + y * 10000
            phaseType = ""
            if timerID in [1]:
                phaseType = "B"
                chromeTraces.append(
                    {"name": f"{risc}", "cat": "PERF", "ph": phaseType, "pid": traceID, "tid": traceID, "ts": minTime}
                )
            elif timerID in [4]:
                phaseType = "E"

                chromeTraces.append(
                    {"name": f"{risc}", "cat": "PERF", "ph": phaseType, "pid": traceID, "tid": traceID, "ts": timestamp}
                )

            chromeTraces.append(
                {"name": "thread_name", "ph": "M", "pid": traceID, "tid": traceID, "args": {"name": f"{core}"}}
            )

    with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceChromeTracing}", "w") as devicesDataJson:
        json.dump(chromeTraces, devicesDataJson, indent=2, cls=TupleEncoder, sort_keys=True)


def print_json(devicesData, setup):
    with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceAnalysisData}", "w") as devicesDataJson:
        json.dump({"data": devicesData, "setup": setup}, devicesDataJson, indent=2, cls=TupleEncoder, sort_keys=True)


def print_rearranged_csv(devicesData, setup, freqText=None):
    timerIDLabels = setup.timerIDLabels[1:5]
    if not freqText:
        freqText = devicesData["deviceInfo"]["freq"]
    with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceRearranged}", "w") as timersCSV:
        for chipID, deviceData in devicesData["devices"].items():
            timeWriter = csv.writer(timersCSV, delimiter=",")

            timeWriter.writerow(["Clock Frequency [MHz]", freqText])
            timeWriter.writerow(["PCIe slot", chipID])
            timeWriter.writerow(
                ["core x", "core y"]
                + [f"BRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
                + [f"NCRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
            )
            for core in sorted(deviceData["cores"].keys(), key=coreCompare):
                if type(core) == tuple:
                    core_x, core_y = core
                    timeWriter.writerow(
                        [core_x, core_y]
                        + return_available_timer("BRISC", deviceData["cores"][core]["riscs"], timerIDLabels)
                        + return_available_timer("NCRISC", deviceData["cores"][core]["riscs"], timerIDLabels)
                    )


def analyze_stats(timerStats, timerStatsCores):
    FW_START_VARIANCE_THRESHOLD = 1e3
    if int(timerStats["FW start"]["Max"]) > FW_START_VARIANCE_THRESHOLD:
        print(f"NOTE: Variance on FW starts seems too high at : {timerStats['FW start']['Max']} [cycles]")
        print(f"Please reboot the host to make sure the device is not in a bad reset state")


def is_print_supported(devicesData):
    return devicesData["deviceInfo"]["arch"] == "grayskull"


def print_stats_outfile(devicesData, setup):
    if is_print_supported(devicesData):
        original_stdout = sys.stdout
        with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceStatsTXT}", "w") as statsFile:
            sys.stdout = statsFile
            print_stats(devicesData, setup)
            sys.stdout = original_stdout


def print_stats(devicesData, setup):
    if not is_print_supported(devicesData):
        print(f"{devicesData['deviceInfo']['arch']} stat print is not supported")
    else:
        numberWidth = 17
        for chipID, deviceData in devicesData["devices"].items():
            for analysis in setup.timerAnalysis.keys():
                if (
                    "analysis" in deviceData["cores"]["DEVICE"].keys()
                    and analysis in deviceData["cores"]["DEVICE"]["analysis"].keys()
                ):
                    assert "stats" in deviceData["cores"]["DEVICE"]["analysis"][analysis].keys()
                    stats = deviceData["cores"]["DEVICE"]["analysis"][analysis]["stats"]
                    print()
                    print(f"=================== {analysis} ===================")
                    if stats["Count"] > 1:
                        for stat in setup.displayStats:
                            if stat in ["Count"]:
                                print(f"{stat:>12}          = {stats[stat]:>10,.0f}")
                            else:
                                print(f"{stat:>12} [cycles] = {stats[stat]:>10,.0f}")
                    else:
                        print(f"{'Duration':>12} [cycles] = {stats['Max']:>10,.0f}")
                    print()
                    if setup.timerAnalysis[analysis]["across"] in ["risc", "core"]:
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
                                        core = (core_x, core_y)
                                        if core_y > 5:
                                            core = (core_x, core_y - 1)
                                        noCoreData = True
                                        if core in deviceData["cores"].keys():
                                            for risc, riscData in deviceData["cores"][core]["riscs"].items():
                                                if (
                                                    "analysis" in riscData.keys()
                                                    and analysis in riscData["analysis"].keys()
                                                ):
                                                    stats = riscData["analysis"][analysis]["stats"]
                                                    plusMinus = (stats["Max"] - stats["Min"]) // 2
                                                    median = stats["Median"]
                                                    tmpStr = f"{median:,.0f}"
                                                    if stats["Count"] > 1:
                                                        tmpStr = "{tmpStr}{sign}{plusMinus:,}".format(
                                                            tmpStr=tmpStr, sign="\u00B1", plusMinus=plusMinus
                                                        )
                                                    print(f"{tmpStr:>{numberWidth}}", end="")
                                                    noCoreData = False
                                        if noCoreData:
                                            print(f"{'X':>{numberWidth}}", end="")
                                    else:
                                        if core_x in [0, 3, 6, 9]:
                                            print(f"{f'DRAM{4 + int(core_x/3)}':>{numberWidth}}", end="")
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
    print("Please choose a plot setup class that matches your test kernel profile data.")
    print("e.g. : process_device_log.py test_add_two_ints")
    print("Or run default by providing no args")
    print("e.g. : process_device_log.py")


def extract_device_info(deviceInfo):
    if "Chip clock is at " in deviceInfo[0]:
        return "grayskull", 1200
    elif "ARCH" in deviceInfo[0]:
        arch = deviceInfo[0].split(":")[-1].strip(" \n")
        freq = deviceInfo[1].split(":")[-1].strip(" \n")
        return arch, int(freq)
    else:
        raise Exception


def import_device_profile_log(
    logPath,
    xRange=None,
    intrestingCores=None,
    ignoreMarkers=None,
):
    devicesData = {"devices": {}}
    programsData = {"programs": {}}
    with open(logPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        arch = ""
        freq = ""
        for lineCount, row in enumerate(csvReader):
            if lineCount == 0:
                arch, freq = extract_device_info(row)
                devicesData.update(dict(deviceInfo=dict(arch=arch, freq=freq)))

            elif lineCount > 1:
                programID = int(row[0].strip())
                chipID = int(row[1])
                core = (int(row[2]), int(row[3]))
                if intrestingCores and core not in intrestingCores:
                    continue
                risc = row[4].strip()
                timerID = int(row[5])
                if ignoreMarkers and timerID in ignoreMarkers:
                    continue
                timeData = int(row[6])

                if chipID in devicesData["devices"].keys():
                    if core in devicesData["devices"][chipID]["cores"].keys():
                        if risc in devicesData["devices"][chipID]["cores"][core]["riscs"].keys():
                            devicesData["devices"][chipID]["cores"][core]["riscs"][risc]["timeseries"].append(
                                (timerID, timeData)
                            )
                        else:
                            devicesData["devices"][chipID]["cores"][core]["riscs"][risc] = {
                                "timeseries": [(timerID, timeData)]
                            }
                    else:
                        devicesData["devices"][chipID]["cores"][core] = {
                            "riscs": {risc: {"timeseries": [(timerID, timeData)]}}
                        }
                else:
                    devicesData["devices"][chipID] = {
                        "cores": {core: {"riscs": {risc: {"timeseries": [(timerID, timeData)]}}}}
                    }

                if programID in programsData["programs"].keys():
                    if chipID in programsData["programs"][programID]["devices"].keys():
                        if core in programsData["programs"][programID]["devices"][chipID]["cores"].keys():
                            if (
                                risc
                                in programsData["programs"][programID]["devices"][chipID]["cores"][core]["riscs"].keys()
                            ):
                                programsData["programs"][programID]["devices"][chipID]["cores"][core]["riscs"][risc][
                                    "timeseries"
                                ].append((timerID, timeData))
                            else:
                                programsData["programs"][programID]["devices"][chipID]["cores"][core]["riscs"][risc] = {
                                    "timeseries": [(timerID, timeData)]
                                }
                        else:
                            programsData["programs"][programID]["devices"][chipID]["cores"][core] = {
                                "riscs": {risc: {"timeseries": [(timerID, timeData)]}}
                            }
                    else:
                        programsData["programs"][programID]["devices"][chipID] = {
                            "cores": {core: {"riscs": {risc: {"timeseries": [(timerID, timeData)]}}}}
                        }
                else:
                    programsData["programs"][programID] = {
                        "devices": {chipID: {"cores": {core: {"riscs": {risc: {"timeseries": [(timerID, timeData)]}}}}}}
                    }
                    programsData["programs"][programID].update(dict(deviceInfo=dict(arch=arch, freq=freq)))

    def sort_timeseries_and_find_min(devicesData):
        globalMinTS = (1 << 64) - 1
        globalMinRisc = "BRISC"
        globalMinCore = (0, 0)

        foundRange = set()
        for chipID, deviceData in devicesData["devices"].items():
            for core, coreData in deviceData["cores"].items():
                for risc, riscData in coreData["riscs"].items():
                    riscData["timeseries"].sort(key=lambda x: x[1])
                    firstTimeID, firsTimestamp = riscData["timeseries"][0]
                    if globalMinTS > firsTimestamp:
                        globalMinTS = firsTimestamp
                        globalMinCore = core
                        globalMinRisc = risc
            deviceData.update(
                dict(metadata=dict(global_min=dict(ts=globalMinTS, risc=globalMinRisc, core=globalMinCore)))
            )

        for chipID, deviceData in devicesData["devices"].items():
            for core, coreData in deviceData["cores"].items():
                for risc, riscData in coreData["riscs"].items():
                    riscData["timeseries"].sort(key=lambda x: x[1])
                    newTimeseries = []
                    for marker, timestamp in riscData["timeseries"]:
                        shiftedTS = timestamp - globalMinTS
                        if xRange and xRange[0] < shiftedTS < xRange[1]:
                            newTimeseries.append((marker, shiftedTS))
                    if newTimeseries:
                        riscData["timeseries"] = newTimeseries
                        foundRange.add((chipID, core, risc))
                    else:
                        riscData["timeseries"].insert(0, (0, deviceData["metadata"]["global_min"]["ts"]))

        if foundRange:
            for chipID, deviceData in devicesData["devices"].items():
                for core, coreData in deviceData["cores"].items():
                    for risc, riscData in coreData["riscs"].items():
                        if (chipID, core, risc) not in foundRange:
                            riscData["timeseries"] = []

    # Sort all timeseries and find global min timestamp
    sort_timeseries_and_find_min(devicesData)
    # for programID, programData in programsData["programs"].items():
    # sort_timeseries_and_find_min(programData)

    return devicesData, programsData


def is_new_op_core(tsRisc):
    timerID, tsValue, risc = tsRisc
    if risc == "BRISC" and timerID == 1:
        return True
    return False


def is_new_op_device(tsCore, coreOpMap):
    timerID, tsValue, risc, core = tsCore
    appendTs = False
    isNewOp = False
    isNewOpFinished = False
    # if core[1] != NON_COMPUTE_ROW:
    if False:
        appendTs = True
        if risc == "BRISC" and timerID == 1:
            assert (
                core not in coreOpMap.keys()
            ), f"Unexpected BRISC start in {tsCore} {coreOpMap[core]}, this could be caused by soft resets"
            isNewOp = True
            coreOpMap[core] = (tsValue,)
        elif risc == "BRISC" and timerID == 4:
            assert core in coreOpMap.keys() and len(coreOpMap[core]) == 1, "Unexpected BRISC end"
            coreOpMap[core] = (coreOpMap[core][0], tsValue)
            isNewOpFinished = True
            for opDuration in coreOpMap.values():
                pairSize = len(opDuration)
                assert pairSize == 1 or pairSize == 2, "Wrong op duration"
                if pairSize == 1:
                    isNewOpFinished = False
                    break
    return appendTs, isNewOp, isNewOpFinished


def risc_to_core_timeseries(devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            tmpTimeseries = []
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    tmpTimeseries.append(ts + (risc,))

            tmpTimeseries.sort(key=lambda x: x[1])

            ops = []
            for ts in tmpTimeseries:
                timerID, tsValue, risc = ts
                if is_new_op_core(ts):
                    ops.append([ts])
                else:
                    if len(ops) > 0:
                        ops[-1].append(ts)

            coreData["riscs"]["TENSIX"] = {"timeseries": tmpTimeseries, "ops": ops}


def core_to_device_timeseries(devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        tmpTimeseries = {"riscs": {}}
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    timerID, timestamp, *metadata = ts
                    tsCore = ts + (core,)
                    if timerID == 0:
                        tsCore = ts + (deviceData["metadata"]["global_min"]["core"],)
                    if risc in tmpTimeseries["riscs"].keys():
                        tmpTimeseries["riscs"][risc]["timeseries"].append(tsCore)
                    else:
                        tmpTimeseries["riscs"][risc] = {"timeseries": [tsCore]}

        for risc in tmpTimeseries["riscs"].keys():
            tmpTimeseries["riscs"][risc]["timeseries"].sort(key=lambda x: x[1])

        ops = []
        coreOpMap = {}
        for ts in tmpTimeseries["riscs"]["TENSIX"]["timeseries"]:
            appendTs, isNewOp, isNewOpFinished = is_new_op_device(ts, coreOpMap)
            if appendTs:
                if isNewOp:
                    ops.append({"timeseries": []})
                ops[-1]["timeseries"].append(ts)
            if isNewOpFinished:
                coreOpMap = {}

        tmpTimeseries["riscs"]["TENSIX"]["ops"] = ops
        deviceData["cores"]["DEVICE"] = tmpTimeseries


def timeseries_to_durations(deviceData):
    for core, coreData in deviceData["cores"].items():
        for risc, riscData in coreData["riscs"].items():
            riscData["durations"] = {"data": {}, "order": []}
            timeseries = riscData["timeseries"]
            for startData, endData in zip(timeseries[:-1], timeseries[1:]):
                startTimerID, startTime, *startMeta = startData
                endTimerID, endTime, *endMeta = endData
                start = startTimerID
                if startMeta:
                    start = (startTimerID,) + tuple(startMeta)
                end = endTimerID
                if endMeta:
                    end = (endTimerID,) + tuple(endMeta)
                durationType = (start, end)
                if durationType in riscData["durations"]["data"].keys():
                    riscData["durations"]["data"][durationType].append((startTime, endTime, endTime - startTime))
                else:
                    riscData["durations"]["data"][durationType] = [(startTime, endTime, endTime - startTime)]
                riscData["durations"]["order"].append(
                    (durationType, len(riscData["durations"]["data"][durationType]) - 1)
                )


def plotData_to_timelineXVals(deviceData, plotCores, setup):
    plotRiscs = setup.riscs
    xValsDict = {risc: [] for risc in plotRiscs}
    traces = {risc: [] for risc in plotRiscs}

    coreOrderTrav = {core: {risc: 0 for risc in deviceData["cores"][core]["riscs"].keys()} for core in plotCores}
    for risc in plotRiscs:
        ordering = True
        traceToAdd = None
        discardedTraces = set()
        while ordering:
            ordering = False
            addTrace = True
            for core in plotCores:
                assert core in deviceData["cores"].keys()
                if risc in deviceData["cores"][core]["riscs"].keys():
                    if coreOrderTrav[core][risc] < len(deviceData["cores"][core]["riscs"][risc]["durations"]["order"]):
                        ordering = True
                        trace = deviceData["cores"][core]["riscs"][risc]["durations"]["order"][
                            coreOrderTrav[core][risc]
                        ]
                        if traceToAdd:
                            if core not in traceToAdd[1]:
                                if traceToAdd[0] == trace:
                                    traceToAdd[1].add(core)
                                else:
                                    # Let see if any trace in the future is the candidate for this core
                                    for i in range(
                                        coreOrderTrav[core][risc] + 1,
                                        len(deviceData["cores"][core]["riscs"][risc]["durations"]["order"]),
                                    ):
                                        futureTrace = deviceData["cores"][core]["riscs"][risc]["durations"]["order"][i]
                                        if futureTrace == traceToAdd[0] and traceToAdd[0] not in discardedTraces:
                                            # Pick a better candidate and put this in discarded so it cannot picked
                                            # again this round. This is to avoid forever loop in the while loop
                                            discardedTraces.add(traceToAdd[0])
                                            traceToAdd = (trace, set([core]))
                                            addTrace = False
                                            break
                                    if addTrace == False:
                                        break
                        else:
                            # Pick a new candidate
                            traceToAdd = (trace, set([core]))
                            addTrace = False
                            break

            if addTrace and traceToAdd:
                if traceToAdd[0] in discardedTraces:
                    discardedTraces.remove(traceToAdd[0])
                traces[risc].append(traceToAdd)
                for core in traceToAdd[1]:
                    if risc in deviceData["cores"][core]["riscs"].keys():
                        coreOrderTrav[core][risc] += 1
                traceToAdd = None

    for risc in traces.keys():
        for trace in traces[risc]:
            xVals = []
            traceType = trace[0]
            cores = trace[1]
            for core in plotCores:
                xVal = 0
                if core in cores:
                    xVal = deviceData["cores"][core]["riscs"][risc]["durations"]["data"][traceType[0]][traceType[1]][2]
                xVals.append(xVal)
            xValsDict[risc].append((traceType, xVals))
    return xValsDict


def timeline_plot(yVals, xValsDict, setup):
    riscsData = setup.riscsData
    timerIDLabels = setup.timerIDLabels

    layout = go.Layout(xaxis=dict(title="Cycle count"))
    if len(yVals) > 1:
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
    for risc in setup.riscs:
        durations = []
        if risc in xValsDict.keys():
            for xVals in xValsDict[risc]:
                (duration, instance), xValsData = xVals
                if duration not in durations:
                    durations.append(duration)

            colors = sns.color_palette(riscsData[risc]["color"], len(durations) + 1).as_hex()
            colorMap = {duration: color for duration, color in zip(durations, colors)}
            colorMap["TRANSPARENT"] = "rgb(135,206,250)"
            colorMap["DARK"] = colors[-1]

            for xVals in xValsDict[risc]:
                (duration, instance), xValsData = xVals
                startData, endData = duration
                if type(startData) == tuple:
                    (start, *startMeta) = startData
                else:
                    start = startData
                    startMeta = None
                if type(endData) == tuple:
                    (end, *endMeta) = endData
                else:
                    end = endData
                    endMeta = None

                if (start, end) in [(4, 1), (0, 1)]:
                    color = colorMap["TRANSPARENT"]
                elif (start, end) in [(1, 2), (3, 4)]:
                    color = colorMap["DARK"]
                else:
                    color = colorMap[duration]

                for timerID, text in timerIDLabels:
                    if start == timerID:
                        start = text
                    if end == timerID:
                        end = text

                startTxt = f"{start}"
                if startMeta:
                    startTxt = f"{start},{startMeta}"
                endTxt = f"{end}"
                if endMeta:
                    endTxt = f"{end},{endMeta}"
                name = f"{startTxt}->{endTxt}"

                showlegend = False

                fig.add_trace(
                    go.Bar(
                        y=[yVals, [risc] * len(yVals)],
                        x=xValsData,
                        orientation="h",
                        name="",
                        showlegend=showlegend,
                        marker=dict(color=color),
                        customdata=[name for i in range(len(xValsData))],
                        hovertemplate="<br>".join(["%{customdata}", "%{x} cycles"]),
                    )
                )
                fig.update_xaxes(
                    showspikes=True,
                    spikecolor="green",
                    spikesnap="cursor",
                    spikemode="across",
                    spikethickness=0.5,
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

    fig.update_layout(barmode="stack", height=setup.plotBaseHeight + setup.plotPerCoreHeight * len(yVals))

    return fig


def translate_metaData(metaData, core, risc):
    metaRisc = None
    metaCore = None
    if len(metaData) == 2:
        metaRisc, metaCore = metaData
    elif len(metaData) == 1:
        content = metaData[0]
        if type(content) == str:
            metaRisc = content
        elif type(content) == tuple:
            metaCore = content
    if core != "ANY" and metaCore:
        core = metaCore
    if risc != "ANY" and metaRisc:
        risc = metaRisc
    return core, risc


def determine_conditions(timerID, metaData, analysis):
    currCore = analysis["start"]["core"] if "core" in analysis["start"].keys() else None
    currRisc = analysis["start"]["risc"]
    currStart = (timerID,) + translate_metaData(metaData, currCore, currRisc)

    currCore = analysis["end"]["core"] if "core" in analysis["end"].keys() else None
    currRisc = analysis["end"]["risc"]
    currEnd = (timerID,) + translate_metaData(metaData, currCore, currRisc)

    desStart = (
        analysis["start"]["timerID"],
        analysis["start"]["core"] if "core" in analysis["start"].keys() else None,
        analysis["start"]["risc"],
    )
    desEnd = (
        analysis["end"]["timerID"],
        analysis["end"]["core"] if "core" in analysis["end"].keys() else None,
        analysis["end"]["risc"],
    )
    return currStart, currEnd, desStart, desEnd


def first_last_analysis(timeseries, analysis):
    durations = []
    startFound = None
    for index, (timerID, timestamp, *metaData) in enumerate(timeseries):
        currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
        if not startFound:
            if currStart == desStart:
                startFound = (index, timerID, timestamp)
                break

    if startFound:
        startIndex, startID, startTS = startFound
        for i in range(len(timeseries) - 1, startIndex, -1):
            timerID, timestamp, *metaData = timeseries[i]
            currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
            if currEnd == desEnd:
                durations.append(
                    dict(start=startTS, end=timestamp, durationType=(startID, timerID), diff=timestamp - startTS)
                )
                break

    return durations


def model_first_last_analysis(riscData, analysis):
    return first_last_analysis(riscData["timeseries"], analysis)


def op_first_last_analysis(riscData, analysis):
    durations = []
    if "ops" in riscData.keys():
        for op in riscData["ops"]:
            durations += first_last_analysis(op, analysis)
    return durations


def adjacent_LF_analysis(riscData, analysis):
    timeseries = riscData["timeseries"]
    durations = []
    startFound = None
    for timerID, timestamp, *metaData in timeseries:
        currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
        if not startFound:
            if currStart == desStart:
                startFound = (timerID, timestamp)
        else:
            if currEnd == desEnd:
                startID, startTS = startFound
                durations.append(
                    dict(start=startTS, end=timestamp, durationType=(startID, timerID), diff=timestamp - startTS)
                )
                startFound = None
            elif currStart == desStart:
                startFound = (timerID, timestamp)

    return durations


def timeseries_analysis(riscData, name, analysis):
    tmpList = []
    if analysis["type"] == "adjacent":
        tmpList = adjacent_LF_analysis(riscData, analysis)
    elif analysis["type"] == "model_first_last":
        tmpList = model_first_last_analysis(riscData, analysis)
    elif analysis["type"] == "op_first_last":
        tmpList = op_first_last_analysis(riscData, analysis)

    tmpDF = pd.DataFrame(tmpList)
    tmpDict = {}
    if not tmpDF.empty:
        tmpDict = {
            "analysis": analysis,
            "stats": {
                "Count": tmpDF.loc[:, "diff"].count(),
                "Average": tmpDF.loc[:, "diff"].mean(),
                "Max": tmpDF.loc[:, "diff"].max(),
                "Min": tmpDF.loc[:, "diff"].min(),
                "Range": tmpDF.loc[:, "diff"].max() - tmpDF.loc[:, "diff"].min(),
                "Median": tmpDF.loc[:, "diff"].median(),
                "Sum": tmpDF.loc[:, "diff"].sum(),
                "First": tmpDF.loc[0, "diff"],
            },
            "series": tmpList,
        }
    if tmpDict:
        if "analysis" not in riscData.keys():
            riscData["analysis"] = {name: tmpDict}
        else:
            riscData["analysis"][name] = tmpDict


def core_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            if core != "DEVICE":
                risc = "TENSIX"
                assert risc in coreData["riscs"].keys()
                riscData = coreData["riscs"][risc]
                timeseries_analysis(riscData, name, analysis)


def device_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"].keys()
        assert risc in deviceData["cores"][core]["riscs"].keys()
        riscData = deviceData["cores"][core]["riscs"][risc]
        timeseries_analysis(riscData, name, analysis)


def ops_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"].keys()
        assert risc in deviceData["cores"][core]["riscs"].keys()
        riscData = deviceData["cores"][core]["riscs"][risc]
        if "ops" in riscData.keys():
            for op in riscData["ops"]:
                timeseries_analysis(op, name, analysis)


def generate_device_level_summary(devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        analysisLists = {}
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                if "analysis" in riscData.keys():
                    for name, analysis in riscData["analysis"].items():
                        if name in analysisLists.keys():
                            analysisLists[name]["statList"].append(analysis["stats"])
                        else:
                            analysisLists[name] = dict(analysis=analysis["analysis"], statList=[analysis["stats"]])

        for name, analysisList in analysisLists.items():
            tmpDF = pd.DataFrame(analysisList["statList"])
            tmpDict = {}
            if not tmpDF.empty:
                tmpDict = {
                    "analysis": analysisList["analysis"],
                    "stats": {
                        "Count": tmpDF.loc[:, "Count"].sum(),
                        "Average": tmpDF.loc[:, "Sum"].sum() / tmpDF.loc[:, "Count"].sum(),
                        "Max": tmpDF.loc[:, "Max"].max(),
                        "Min": tmpDF.loc[:, "Min"].min(),
                        "Range": tmpDF.loc[:, "Max"].max() - tmpDF.loc[:, "Min"].min(),
                        "Median": tmpDF.loc[:, "Median"].median(),
                        "Sum": tmpDF.loc[:, "Sum"].sum(),
                    },
                }
            if "analysis" in deviceData["cores"]["DEVICE"].keys():
                deviceData["cores"]["DEVICE"]["analysis"][name] = tmpDict
            else:
                deviceData["cores"]["DEVICE"]["analysis"] = {name: tmpDict}


def validate_setup(ctx, param, setup):
    setups = []
    for name, obj in inspect.getmembers(device_post_proc_config):
        if inspect.isclass(obj):
            setups.append(name)
    if setup not in setups:
        raise click.BadParameter(f"Setup {setup} not available")
    return getattr(device_post_proc_config, setup)()


def import_log_run_stats(setup=device_post_proc_config.default_setup()):
    devicesData, programsData = import_device_profile_log(
        setup.deviceInputLog, setup.cycleRange, setup.intrestingCores, setup.ignoreMarkers
    )
    risc_to_core_timeseries(devicesData)
    core_to_device_timeseries(devicesData)

    for name, analysis in sorted(setup.timerAnalysis.items()):
        if analysis["across"] == "core":
            core_analysis(name, analysis, devicesData)
        elif analysis["across"] == "device":
            device_analysis(name, analysis, devicesData)
        elif analysis["across"] == "ops":
            ops_analysis(name, analysis, devicesData)

    generate_device_level_summary(devicesData)
    return devicesData


def generate_plots(devicesData, setup, saveFigure=True):
    timelineFigs = {}
    for chipID, deviceData in devicesData["devices"].items():
        timeseries_to_durations(deviceData)
        yVals = sorted(deviceData["cores"].keys(), key=coreCompare, reverse=True)
        yVals.remove("DEVICE")

        xValsDict = plotData_to_timelineXVals(deviceData, yVals, setup)
        key = f"Chip {chipID} Cores"
        timelineFigs[key] = timeline_plot(yVals, xValsDict, setup)

        figHtmls = {
            f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{fig.replace(' ','_')}_{setup.devicePerfHTML}": fig
            for fig in sorted(timelineFigs.keys())
        }

        if saveFigure:
            for filename, figHtml in figHtmls.items():
                timelineFigs[figHtml].write_html(filename)

    return timelineFigs


def run_dashbaord_webapp(devicesData, timelineFigs, setup):
    statTables = {}
    for chipID, deviceData in devicesData["devices"].items():
        key = f"Chip {chipID} Cores"
        if "analysis" in deviceData["cores"]["DEVICE"].keys():
            statTables[key] = generate_analysis_table(deviceData["cores"]["DEVICE"]["analysis"], setup)
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    plotsDiv = []
    for num, item in enumerate(sorted(set(timelineFigs.keys()) | set(statTables.keys()))):
        plotRiscs = set()
        for marker in timelineFigs[item]["data"]:
            if marker["y"][-1][0] in setup.riscs:
                plotRiscs.add(marker["y"][-1][0])

        plotsDiv += [
            html.Div(
                [
                    html.H5(f"{item}:"),
                    statTables[item] if item in statTables.keys() else html.Div([]),
                    html.Br(),
                    html.Br(),
                    html.P("X Axix select box diff [Cycles]: ", style={"display": "inline"}),
                    html.P("", id=f"selected-data-{num}", style={"display": "inline"}),
                    dcc.Graph(id=f"figure-{num}", figure=timelineFigs[item])
                    if item in timelineFigs.keys()
                    else html.Div([]),
                    dcc.Store(
                        id=f"figure-height-{num}",
                        data=dict(
                            height=timelineFigs[item]["layout"]["height"],
                            perRiscHeight=(setup.plotPerCoreHeight) / len(plotRiscs),
                            baseHeight=setup.plotBaseHeight,
                        ),
                    ),
                ]
            )
        ]

    app.layout = html.Div(
        [html.H1("Device Profiler Dashboard", id="main-header")]
        + [
            html.Button("Refresh", id="btn-refresh", style={"margin-right": "15px"}),
            html.Button("Download Artifacts", id="btn-download-artifacts", style={"margin-right": "15px"}),
            dcc.Download(id="download-artifacts"),
            html.P("", id="p-download-message-bar", style={"display": "inline"}),
            html.Br(),
            html.Br(),
        ]
        + plotsDiv
    )

    for num, item in enumerate(sorted(set(timelineFigs.keys()) | set(statTables.keys()))):
        app.clientside_callback(
            """
            function (layoutData, fig, layoutDefaults) {
                newFig = JSON.parse(JSON.stringify(fig))
                if (layoutData['yaxis.autorange'] !== 'undefined' && layoutData['yaxis.autorange'] === true)
                {
                    newFig['layout']['height'] = layoutDefaults['height']
                }
                else if (layoutData['yaxis.range[0]'] !== 'undefined')
                {
                    newFig['layout']['height'] = layoutDefaults['perRiscHeight']*
                        (layoutData['yaxis.range[1]'] - layoutData['yaxis.range[0]'])
                        + layoutDefaults['baseHeight']
                }
                return newFig
            }
            """,
            Output(f"figure-{num}", "figure"),
            [Input(f"figure-{num}", "relayoutData")],  # this triggers the event
            [State(f"figure-{num}", "figure"), State(f"figure-height-{num}", "data")],
            prevent_initial_call=True,
        )

        app.clientside_callback(
            """
            function (selectedData) {
                if (selectedData !== null && selectedData.hasOwnProperty('range') &&  selectedData.range.hasOwnProperty('x'))
                {
                    return (selectedData.range.x[1] - selectedData.range.x[0]).toFixed(0)
                }
                else
                {
                    return ""
                }
            }
            """,
            Output(f"selected-data-{num}", "children"),
            [Input(f"figure-{num}", "selectedData")],  # this triggers the event
            prevent_initial_call=True,
        )

    # @app.callback(
    # Output('selected-data', 'children'),
    # Input('basic-interactions', 'selectedData'))
    # def display_selected_data(selectedData):
    # return json.dumps(selectedData, indent=2)

    @app.callback(Output("btn-refresh", "children"), Input("btn-refresh", "n_clicks"), prevent_initial_call=True)
    def refresh_callback(n_clicks):
        os.system("touch dummy_refresh.py")
        return "Refreshing ..."

    @app.callback(
        Output("p-download-message-bar", "children"),
        Output("download-artifacts", "data"),
        Input("btn-download-artifacts", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_callback(n_clicks):
        newTarballName = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{setup.deviceTarball}"
        ret = os.system(
            f"cd {PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}; mv {setup.deviceTarball} {newTarballName} > /dev/null 2>&1"
        )
        if ret == 0:
            return "", dcc.send_file(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{newTarballName}")
        return "No artifact tarball found, make sure webapp is started without the --no-artifact flag", None

    app.run_server(host="0.0.0.0", port=setup.webappPort, debug=True)


def prepare_output_folder(setup):
    os.system(
        f"rm -rf {PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}; mkdir -p {PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}; cp {setup.deviceInputLog} {PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}"
    )


def generate_artifact_tarball(setup):
    os.system(
        f"cd {PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}; tar -czf ../{setup.deviceTarball} .; mv ../{setup.deviceTarball} ."
    )


@click.command()
@click.option("-s", "--setup", default="default_setup", callback=validate_setup, help="Post processing configurations")
@click.option(
    "-d", "--device-input-log", type=click.Path(exists=True, dir_okay=False), help="Input device side csv log"
)
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for plots and stats")
@click.option("-p", "--port", type=int, help="Dashboard webapp port")
@click.option("--no-print-stats", default=False, is_flag=True, help="Do not print timeline stats")
@click.option("--no-webapp", default=False, is_flag=True, help="Do not run profiler dashboard webapp")
@click.option("--no-plots", default=False, is_flag=True, help="Do not generate plots")
@click.option("--no-artifacts", default=False, is_flag=True, help="Do not generate artifacts tarball")
def main(setup, device_input_log, output_folder, port, no_print_stats, no_webapp, no_plots, no_artifacts):
    if device_input_log:
        setup.deviceInputLog = device_input_log
    if output_folder:
        setup.outputFolder = output_folder
    if port:
        setup.webappPort = port

    devicesData = import_log_run_stats(setup)

    prepare_output_folder(setup)

    # print_stats_outfile(devicesData, setup)
    # print_rearranged_csv(devicesData, setup)
    print_json(devicesData, setup)

    if not no_print_stats:
        print_stats(devicesData, setup)
        for opCount, op in enumerate(devicesData["devices"][0]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]):
            if "analysis" in op.keys():
                print(f"{opCount +1} : {op['analysis']['OPs']['series']}")

    timelineFigs = {}
    if not no_plots:
        timelineFigs = generate_plots(devicesData, setup)

    # if not no_artifacts:
    # generate_artifact_tarball(setup)

    if not no_webapp:
        run_dashbaord_webapp(devicesData, timelineFigs, setup)


if __name__ == "__main__":
    main()
