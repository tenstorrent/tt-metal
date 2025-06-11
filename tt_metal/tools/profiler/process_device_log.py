#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import inspect
import csv
import json
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import click
from loguru import logger

from tt_metal.tools.profiler.common import PROFILER_ARTIFACTS_DIR
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

SUM_MARKER_ID_START = 3000

dispatchCores = set()


def coreCompare(core):
    if type(core) == str:
        return (1 << 64) - 1
    x = core[0]
    y = core[1]
    return x + y * 100


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


def print_json(devicesData, setup):
    with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceAnalysisData}", "w") as devicesDataJson:
        json.dump({"data": devicesData, "setup": setup}, devicesDataJson, indent=2, cls=TupleEncoder, sort_keys=True)


def analyze_stats(timerStats, timerStatsCores):
    FW_START_VARIANCE_THRESHOLD = 1e3
    if int(timerStats["FW start"]["Max"]) > FW_START_VARIANCE_THRESHOLD:
        print(f"NOTE: Variance on FW starts seems too high at : {timerStats['FW start']['Max']} [cycles]")
        print(f"Please reboot the host to make sure the device is not in a bad reset state")


def print_stats_outfile(devicesData, setup):
    original_stdout = sys.stdout
    with open(f"{PROFILER_ARTIFACTS_DIR}/{setup.outputFolder}/{setup.deviceStatsTXT}", "w") as statsFile:
        sys.stdout = statsFile
        print_stats(devicesData, setup)
        sys.stdout = original_stdout


def print_stats(devicesData, setup):
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


def print_help():
    print("Please choose a postprocessing config for profile data.")
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


def import_device_profile_log(logPath):
    devicesData = {"devices": {}}
    with open(logPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        arch = ""
        freq = ""
        for lineCount, row in enumerate(csvReader):
            if lineCount == 0:
                arch, freq = extract_device_info(row)
                devicesData.update(dict(deviceInfo=dict(arch=arch, freq=freq)))

            elif lineCount > 1:
                chipID = int(row[0].strip())
                core = (int(row[1].strip()), int(row[2].strip()))
                risc = row[3].strip()
                timerID = {"id": int(row[4].strip()), "zone_name": "", "type": "", "src_line": "", "src_file": ""}
                timeData = int(row[5].strip())
                attachedData = 0
                if len(row) == 13:
                    attachedData = int(row[6].strip())
                    timerID["run_host_id"] = int(row[7].strip())
                    timerID["zone_name"] = row[8].strip()
                    timerID["type"] = row[9].strip()
                    timerID["src_line"] = int(row[10].strip())
                    timerID["src_file"] = row[11].strip()
                    timerID["meta_data"] = row[12].strip()

                if chipID in devicesData["devices"].keys():
                    if core in devicesData["devices"][chipID]["cores"].keys():
                        if risc in devicesData["devices"][chipID]["cores"][core]["riscs"].keys():
                            devicesData["devices"][chipID]["cores"][core]["riscs"][risc]["timeseries"].append(
                                (timerID, timeData, attachedData)
                            )
                        else:
                            devicesData["devices"][chipID]["cores"][core]["riscs"][risc] = {
                                "timeseries": [(timerID, timeData, attachedData)]
                            }
                    else:
                        devicesData["devices"][chipID]["cores"][core] = {
                            "riscs": {risc: {"timeseries": [(timerID, timeData, attachedData)]}}
                        }
                else:
                    devicesData["devices"][chipID] = {
                        "cores": {core: {"riscs": {risc: {"timeseries": [(timerID, timeData, attachedData)]}}}}
                    }

    def sort_timeseries_and_find_min(devicesData):
        globalMinTS = (1 << 64) - 1
        globalMinRisc = "BRISC"
        globalMinCore = (0, 0)

        foundRange = set()
        for chipID, deviceData in devicesData["devices"].items():
            for core, coreData in deviceData["cores"].items():
                for risc, riscData in coreData["riscs"].items():
                    riscData["timeseries"].sort(key=lambda x: x[1])
                    firstTimeID, firsTimestamp, firstStatData = riscData["timeseries"][0]
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
                    for marker, timestamp, attachedData in riscData["timeseries"]:
                        shiftedTS = timestamp - globalMinTS
                        # ERISC dispatch is EOL, some models still use it. Need to check and drop it here until it is fully removed.
                        if (
                            "CQ-DISPATCH" in marker["zone_name"] or "CQ-PREFETCH" in marker["zone_name"]
                        ) and "ERISC" not in risc:
                            dispatchCores.add((chipID, core, risc))
                    riscData["timeseries"].insert(
                        0,
                        (
                            {"id": 0, "zone_name": "", "type": "", "src_line": "", "src_file": ""},
                            deviceData["metadata"]["global_min"]["ts"],
                            0,
                        ),
                    )

        if foundRange:
            for chipID, deviceData in devicesData["devices"].items():
                for core, coreData in deviceData["cores"].items():
                    for risc, riscData in coreData["riscs"].items():
                        if (chipID, core, risc) not in foundRange:
                            riscData["timeseries"] = []

    # Sort all timeseries and find global min timestamp
    sort_timeseries_and_find_min(devicesData)

    return devicesData


def get_ops(timeseries):
    opsDict = {}
    for ts in timeseries:
        timerID, *_ = ts
        if "run_host_id" in timerID:
            opID = timerID["run_host_id"]
            if opID not in opsDict:
                opsDict[opID] = [ts]
            else:
                opsDict[opID].append(ts)

    ordered_ops = list(opsDict.keys())
    # sort over timestamps
    ordered_ops.sort(key=lambda x: opsDict[x][0][1])

    ops = []

    ops.append({"timeseries": []})
    for opID in ordered_ops:
        if opID == 0:
            continue
        op = opsDict[opID]
        opCores = {}

        op.sort(key=lambda ts: ts[1])
        for ts in op:
            if len(ts) == 5:
                timerID, tsValue, attachedData, risc, core = ts
                opCores[core] = None

        for ts in op:
            timerID, *_ = ts
            if timerID["id"] == 0:
                continue
            opIsDone = False
            if len(ts) == 5:
                timerID, tsValue, attachedData, risc, core = ts
                if opCores[core]:
                    if (risc == "BRISC" and timerID["zone_name"] == "BRISC-FW" and timerID["type"] == "ZONE_START") or (
                        risc == "ERISC" and timerID["zone_name"] == "ERISC-FW" and timerID["type"] == "ZONE_START"
                    ):
                        if len(opCores[core]) == 2:
                            corruption = False
                            for core, coreOp in opCores.items():
                                if coreOp and len(coreOp) != 2:
                                    corruption = True
                            if corruption:
                                assertMsg = f"This is before other cores are finished with this op. Data corruption could be the cause of this. Please retry your run"
                            else:
                                assertMsg = f"This is before other cores have reported any activity on this op. Other cores might have their profiler buffer filled up. "
                                assertMsg += "Please either decrease the number of ops being profiled or run dump device profiler more often"
                        else:
                            assertMsg = f"This is before a FW end was received for this op. Data corruption could be the cause of this. Please retry your run"
                        assert (
                            False
                        ), f"Unexpected FW start, core {core}, risc {risc} is reporting a second start of FW for op {opID}. {assertMsg}"

                    elif (risc == "BRISC" and timerID["zone_name"] == "BRISC-FW" and timerID["type"] == "ZONE_END") or (
                        risc == "ERISC" and timerID["zone_name"] == "ERISC-FW" and timerID["type"] == "ZONE_END"
                    ):
                        assert (
                            len(opCores[core]) == 1
                        ), "Unexpected FW end, core {core}, risc {risc} is reporting a second end of FW for op {opID}"
                        opCores[core] = (opCores[core][0], timerID)
                        opIsDone = True
                        for core, coreOp in opCores.items():
                            if not coreOp or len(coreOp) != 2:
                                opIsDone = False
                                break
                else:
                    if (risc == "BRISC" and timerID["zone_name"] == "BRISC-FW" and timerID["type"] == "ZONE_START") or (
                        risc == "ERISC" and timerID["zone_name"] == "ERISC-FW" and timerID["type"] == "ZONE_START"
                    ):
                        opCores[core] = (timerID,)
            if len(ts) == 4:
                timerID, tsValue, attachedData, risc = ts
                if (risc == "BRISC" and timerID["zone_name"] == "BRISC-FW" and timerID["type"] == "ZONE_END") or (
                    risc == "ERISC" and timerID["zone_name"] == "ERISC-FW" and timerID["type"] == "ZONE_END"
                ):
                    opIsDone = True
            ops[-1]["timeseries"].append(ts)
            if opIsDone:
                ops.append({"timeseries": []})
                for core in opCores:
                    opCores[core] = None
    ops.pop()
    return ops


def get_dispatch_core_ops(timeseries):
    masterRisc = "BRISC"
    subordinateRisc = "NCRISC"
    riscData = {
        masterRisc: {"zone": [], "opID": 0, "cmdType": "", "ops": {}, "orderedOpIDs": [], "opFinished": False},
        subordinateRisc: {"zone": [], "opID": 0, "cmdType": "", "ops": {}, "orderedOpIDs": [], "opFinished": False},
    }
    for ts in timeseries:
        timerID, tsValue, attachedData, risc = ts
        riscData[risc]["zone"].append(ts)

        if "meta_data" in timerID and "workers_runtime_id" in timerID["meta_data"]:
            riscData[risc]["opFinished"] = False
            riscData[risc]["opID"] = eval(timerID["meta_data"])["workers_runtime_id"]
            # Only record first trace
            if riscData[risc]["opID"] in riscData[risc]["ops"].keys():
                riscData[risc]["opID"] = 0

        if "meta_data" in timerID and "dispatch_command_type" in timerID["meta_data"]:
            riscData[risc]["cmdType"] = eval(timerID["meta_data"])["dispatch_command_type"]
            if "CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL" in riscData[risc]["cmdType"]:
                riscData[risc]["opFinished"] = True

            if "CQ_DISPATCH_CMD_SEND_GO_SIGNAL" in riscData[risc]["cmdType"]:
                riscData[risc]["opID"] += 1

        if "type" in timerID and timerID["type"] == "ZONE_END":
            riscData[risc]["zone"][0][0]["zone_name"] = riscData[risc]["cmdType"]
            riscData[risc]["zone"][-1][0]["zone_name"] = riscData[risc]["cmdType"]
            if riscData[risc]["opID"] not in riscData[risc]["ops"]:
                riscData[risc]["ops"][riscData[risc]["opID"]] = riscData[risc]["zone"].copy()
            else:
                riscData[risc]["ops"][riscData[risc]["opID"]] += riscData[risc]["zone"]
            riscData[risc]["zone"] = []
            if riscData[risc]["opFinished"]:
                riscData[risc]["opID"] = 0

    for risc, data in riscData.items():
        data["orderedOpIDs"] = list(data["ops"].keys())
        # sort over timestamps
        data["orderedOpIDs"].sort(key=lambda x: data["ops"][x][0][1])

    opsDict = {}
    for masterOpID, subordinateOpID in zip(
        riscData[masterRisc]["orderedOpIDs"], riscData[subordinateRisc]["orderedOpIDs"]
    ):
        opsDict[masterOpID] = (
            riscData[masterRisc]["ops"][masterOpID] + riscData[subordinateRisc]["ops"][subordinateOpID]
        )

        opsDict[masterOpID].sort(key=lambda x: x[1])

    ops = []
    for opID in riscData[masterRisc]["orderedOpIDs"]:
        # opID of zero is non-associated with any op and should be discarded
        if opID > 0:
            ops.append({"timeseries": opsDict[opID]})

    return ops


def get_dispatch_core_ops_core_to_device(chipID, deviceData):
    deviceDispatchCores = set()
    for chip, core, risc in dispatchCores:
        if chip == chipID:
            deviceDispatchCores.add(core)

    ops = []
    for core in deviceDispatchCores:
        for op in deviceData["cores"][core]["riscs"]["TENSIX"]["ops"]:
            ops.append(op)

    ops.sort(key=lambda x: x["timeseries"][0][1])

    return ops


def risc_to_core_timeseries(devicesData, detectOps):
    dispatchCoresNoRisc = set()
    for chip, core, risc in dispatchCores:
        dispatchCoresNoRisc.add((chip, core))

    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            tmpTimeseries = []
            tmpDurations = []
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    tmpTimeseries.append(ts + (risc,))

            tmpTimeseries.sort(key=lambda x: x[1])

            ops = []
            if detectOps:
                if (chipID, core) in dispatchCoresNoRisc:
                    ops = get_dispatch_core_ops(tmpTimeseries)
                else:
                    ops = get_ops(tmpTimeseries)

            coreData["riscs"]["TENSIX"] = {"timeseries": tmpTimeseries, "ops": ops}


def core_to_device_timeseries(devicesData, detectOps):
    for chipID, deviceData in devicesData["devices"].items():
        logger.info(f"Importing Data For Device Number : {chipID}")
        tmpTimeseries = {"riscs": {}}
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    timerID, timestamp, *metadata = ts
                    tsCore = ts + (core,)
                    if timerID["id"] == 0:
                        tsCore = ts + (deviceData["metadata"]["global_min"]["core"],)
                    if risc in tmpTimeseries["riscs"].keys():
                        tmpTimeseries["riscs"][risc]["timeseries"].append(tsCore)
                    else:
                        tmpTimeseries["riscs"][risc] = {"timeseries": [tsCore]}

        for risc in tmpTimeseries["riscs"].keys():
            tmpTimeseries["riscs"][risc]["timeseries"].sort(key=lambda x: x[1])

        tmpTimeseries["riscs"]["TENSIX"]["ops"] = []
        tmpTimeseries["riscs"]["TENSIX"]["dispatch_ops"] = []
        if detectOps:
            dispatchOps = get_dispatch_core_ops_core_to_device(chipID, deviceData)
            tmpTimeseries["riscs"]["TENSIX"]["dispatch_ops"] = dispatchOps

            ops = get_ops(tmpTimeseries["riscs"]["TENSIX"]["timeseries"])
            tmpTimeseries["riscs"]["TENSIX"]["ops"] = ops

        deviceData["cores"]["DEVICE"] = tmpTimeseries


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
    currPhase = (timerID["type"],) if "zone_phase" in analysis["start"].keys() else (None,)
    currStart = (timerID["zone_name"],) + currPhase + translate_metaData(metaData, currCore, currRisc)

    currCore = analysis["end"]["core"] if "core" in analysis["end"].keys() else None
    currRisc = analysis["end"]["risc"]
    currPhase = (timerID["type"],) if "zone_phase" in analysis["end"].keys() else (None,)
    currEnd = (timerID["zone_name"],) + currPhase + translate_metaData(metaData, currCore, currRisc)

    if type(analysis["start"]["zone_name"]) == list:
        desStart = [
            (
                zoneName,
                analysis["start"]["zone_phase"] if "zone_phase" in analysis["start"].keys() else None,
                analysis["start"]["core"] if "core" in analysis["start"].keys() else None,
                analysis["start"]["risc"],
            )
            for zoneName in analysis["start"]["zone_name"]
        ]
    else:
        desStart = [
            (
                analysis["start"]["zone_name"],
                analysis["start"]["zone_phase"] if "zone_phase" in analysis["start"].keys() else None,
                analysis["start"]["core"] if "core" in analysis["start"].keys() else None,
                analysis["start"]["risc"],
            )
        ]

    if type(analysis["end"]["zone_name"]) == list:
        desEnd = [
            (
                zoneName,
                analysis["end"]["zone_phase"] if "zone_phase" in analysis["end"].keys() else None,
                analysis["end"]["core"] if "core" in analysis["end"].keys() else None,
                analysis["end"]["risc"],
            )
            for zoneName in analysis["end"]["zone_name"]
        ]
    else:
        desEnd = [
            (
                analysis["end"]["zone_name"],
                analysis["end"]["zone_phase"] if "zone_phase" in analysis["end"].keys() else None,
                analysis["end"]["core"] if "core" in analysis["end"].keys() else None,
                analysis["end"]["risc"],
            )
        ]

    return currStart, currEnd, desStart, desEnd


def currMark_in_desMarks(currMark, desMarks):
    ret = False
    currName, *curr_ = currMark
    for desMark in desMarks:
        desName, *des_ = desMark
        if des_ == curr_:
            if desName == currName:
                ret = True
                break
            elif "*" == desName[-1]:
                if desName[:-1] in currName:
                    ret = True
                    break
    return ret


def first_last_analysis(timeseries, analysis):
    durations = []
    startFound = None
    for index, (timerID, timestamp, attachedData, *metaData) in enumerate(timeseries):
        currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
        if not startFound:
            if currMark_in_desMarks(currStart, desStart):
                startFound = (index, timerID, timestamp)
                break

    if startFound:
        startIndex, startID, startTS = startFound
        for i in range(len(timeseries) - 1, startIndex, -1):
            timerID, timestamp, attachedData, *metaData = timeseries[i]
            currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
            if currMark_in_desMarks(currEnd, desEnd):
                durations.append(
                    dict(
                        start_cycle=startTS,
                        end_cycle=timestamp,
                        duration_type=(startID, timerID),
                        duration_cycles=timestamp - startTS,
                    )
                )
                break
    return durations


def session_first_last_analysis(riscData, analysis):
    return first_last_analysis(riscData["timeseries"], analysis)


def op_first_last_analysis(riscData, analysis):
    return first_last_analysis(riscData["timeseries"], analysis)


def op_core_first_last_analysis(riscData, analysis):
    core_ops = {}
    durations = []
    for ts in riscData["timeseries"]:
        assert len(ts) == 5
        core = ts[4]
        if core in core_ops:
            core_ops[core].append(ts)
        else:
            core_ops[core] = [ts]
    for core, timeseries in core_ops.items():
        durations.append(first_last_analysis(timeseries, analysis)[0])

    return durations


def get_duration(riscData, analysis):
    totalDuration = 0
    for index, (timerID, timestamp, attachedData, risc, core) in enumerate(riscData["timeseries"]):
        desMarker = {"risc": risc, "zone_name": timerID["zone_name"]}
        if desMarker == analysis["marker"]:
            totalDuration += attachedData
    if totalDuration:
        return [dict(duration_type=analysis["marker"], duration_cycles=totalDuration)]
    return []


def is_timer_id_iteration_start(timerID):
    ret = False
    if timerID["type"] == "ZONE_START" and timerID["zone_name"] == "BRISC-FW":
        ret = True
    return ret


def adjacent_LF_analysis(riscData, analysis):
    timeseries = riscData["timeseries"]
    durations = []
    startFound = None
    startIterMark = None
    iterMark = None
    for timerID, timestamp, attachedData, *metaData in timeseries:
        if is_timer_id_iteration_start(timerID):
            iterMark = (timerID, timestamp)
        currStart, currEnd, desStart, desEnd = determine_conditions(timerID, metaData, analysis)
        if not startFound:
            if currStart in desStart:
                startFound = (timerID, timestamp)
                startIterMark = iterMark
        else:
            if currEnd in desEnd:
                startID, startTS = startFound
                durations.append(
                    dict(
                        start_cycle=startTS,
                        end_cycle=timestamp,
                        duration_type=(startID, timerID),
                        duration_cycles=timestamp - startTS,
                        end_iter_mark=iterMark,
                        start_iter_mark=startIterMark,
                    )
                )
                startFound = None
            elif currStart in desStart:
                startFound = (timerID, timestamp)
                startIterMark = iterMark

    return durations


def timeseries_analysis(riscData, name, analysis):
    tmpList = []
    if analysis["type"] == "adjacent":
        tmpList = adjacent_LF_analysis(riscData, analysis)
    elif analysis["type"] == "session_first_last":
        tmpList = session_first_last_analysis(riscData, analysis)
    elif analysis["type"] == "op_first_last":
        tmpList = op_first_last_analysis(riscData, analysis)
    elif analysis["type"] == "op_core_first_last":
        tmpList = op_core_first_last_analysis(riscData, analysis)
    elif analysis["type"] == "sum":
        tmpList = get_duration(riscData, analysis)
    else:
        return

    tmpDF = pd.DataFrame(tmpList)
    tmpDict = {}
    if not tmpDF.empty:
        tmpDict = {
            "analysis": analysis,
            "stats": {
                "Count": tmpDF.loc[:, "duration_cycles"].count(),
                "Average": tmpDF.loc[:, "duration_cycles"].mean(),
                "Max": tmpDF.loc[:, "duration_cycles"].max(),
                "Min": tmpDF.loc[:, "duration_cycles"].min(),
                "Range": tmpDF.loc[:, "duration_cycles"].max() - tmpDF.loc[:, "duration_cycles"].min(),
                "Median": tmpDF.loc[:, "duration_cycles"].median(),
                "Sum": tmpDF.loc[:, "duration_cycles"].sum(),
                "First": tmpDF.loc[0, "duration_cycles"],
            },
            "series": tmpList,
        }
    if tmpDict:
        if "analysis" not in riscData.keys():
            riscData["analysis"] = {name: tmpDict}
        else:
            riscData["analysis"][name] = tmpDict


def timeseries_events(riscData, name, analysis):
    if analysis["type"] == "event":
        if "events" not in riscData.keys():
            riscData["events"] = {name: []}
        else:
            riscData["events"][name] = []

        for index, (timerID, timestamp, attachedData, risc, *_) in enumerate(riscData["timeseries"]):
            if (timerID["type"] == "TS_EVENT" or timerID["type"] == "TS_DATA") and (
                risc == analysis["marker"]["risc"] or analysis["marker"]["risc"] == "ANY"
            ):
                riscData["events"][name].append((timerID, timestamp, attachedData, risc, *_))


def core_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            if core != "DEVICE":
                risc = "TENSIX"
                assert risc in coreData["riscs"].keys()
                riscData = coreData["riscs"][risc]
                timeseries_analysis(riscData, name, analysis)
                timeseries_events(riscData, name, analysis)


def device_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"].keys()
        assert risc in deviceData["cores"][core]["riscs"].keys()
        riscData = deviceData["cores"][core]["riscs"][risc]
        timeseries_analysis(riscData, name, analysis)
        timeseries_events(riscData, name, analysis)


def ops_analysis(name, analysis, devicesData, doDispatch=False):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"].keys()
        assert risc in deviceData["cores"][core]["riscs"].keys()
        riscData = deviceData["cores"][core]["riscs"][risc]
        if not doDispatch and "ops" in riscData.keys():
            for op in riscData["ops"]:
                timeseries_analysis(op, name, analysis)
                timeseries_events(op, name, analysis)

        elif doDispatch and "dispatch_ops" in riscData.keys():
            for op in riscData["dispatch_ops"]:
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
                if core == "DEVICE" and risc == "TENSIX":
                    if "ops" in riscData.keys():
                        for op in riscData["ops"]:
                            if "analysis" in op.keys():
                                for name, analysis in op["analysis"].items():
                                    if name in analysisLists.keys():
                                        analysisLists[name]["statList"].append(analysis["stats"])
                                    else:
                                        analysisLists[name] = dict(
                                            analysis=analysis["analysis"], statList=[analysis["stats"]]
                                        )
                    if "dispatch_ops" in riscData.keys():
                        for op in riscData["dispatch_ops"]:
                            if "analysis" in op.keys():
                                for name, analysis in op["analysis"].items():
                                    if name in analysisLists.keys():
                                        analysisLists[name]["statList"].append(analysis["stats"])
                                    else:
                                        analysisLists[name] = dict(
                                            analysis=analysis["analysis"], statList=[analysis["stats"]]
                                        )

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
    devicesData = import_device_profile_log(setup.deviceInputLog)

    risc_to_core_timeseries(devicesData, setup.detectOps)
    core_to_device_timeseries(devicesData, setup.detectOps)

    for name, analysis in sorted(setup.timerAnalysis.items()):
        if analysis["across"] == "core":
            core_analysis(name, analysis, devicesData)
        elif analysis["across"] == "device":
            device_analysis(name, analysis, devicesData)
        elif analysis["across"] == "ops":
            ops_analysis(name, analysis, devicesData)
        elif analysis["across"] == "dispatch_ops":
            ops_analysis(name, analysis, devicesData, doDispatch=True)

    generate_device_level_summary(devicesData)
    return devicesData


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
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("--no-print-stats", default=False, is_flag=True, help="Do not print timeline stats")
@click.option("--no-artifacts", default=False, is_flag=True, help="Do not generate artifacts tarball")
@click.option("--no-op-detection", default=False, is_flag=True, help="Do not attempt to detect ops")
def main(setup, device_input_log, output_folder, no_print_stats, no_artifacts, no_op_detection):
    if device_input_log:
        setup.deviceInputLog = device_input_log
    if output_folder:
        setup.outputFolder = output_folder
    if no_op_detection:
        setup.detectOps = False

    devicesData = import_log_run_stats(setup)

    prepare_output_folder(setup)

    print_stats_outfile(devicesData, setup)
    print_json(devicesData, setup)

    if not no_print_stats:
        print_stats(devicesData, setup)

    if not no_artifacts:
        generate_artifact_tarball(setup)


if __name__ == "__main__":
    main()
