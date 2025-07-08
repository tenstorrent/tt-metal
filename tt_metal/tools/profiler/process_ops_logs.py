#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Debug shebang
#!/usr/bin/env -S python3 -m pdb

import os
import csv
from pathlib import Path
import json
import yaml
from datetime import datetime
import copy
from collections import deque

import click
from loguru import logger

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import (
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_HOST_SIDE_LOG,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_OUTPUT_DIR,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
    generate_logs_folder,
    generate_reports_folder,
)

yaml.SafeDumper.ignore_aliases = lambda *args: True

TRACE_OP_ID_BITSHIFT = 32

OUT_NAME = "ops_perf_results"
PER_CORE_OP_TO_OP_OUT_NAME = "per_core_op_to_op_times"
PROFILER_OP_TO_OP_OVERHEAD_NANO_SEC = 1500

OPS_CSV_HEADER = [
    "OP CODE",
    "OP TYPE",
    "GLOBAL CALL COUNT",
    "DEVICE ID",
    "ATTRIBUTES",
    "MATH FIDELITY",
    "CORE COUNT",
    "PARALLELIZATION STRATEGY",
    "HOST START TS",
    "HOST END TS",
    "HOST DURATION [ns]",
    "DEVICE FW START CYCLE",
    "DEVICE FW END CYCLE",
    "OP TO OP LATENCY [ns]",
    "OP TO OP LATENCY BR/NRISC START [ns]",
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE KERNEL DURATION DM START [ns]",
    "DEVICE KERNEL DURATION PER CORE MIN [ns]",
    "DEVICE KERNEL DURATION PER CORE MAX [ns]",
    "DEVICE KERNEL DURATION PER CORE AVG [ns]",
    "DEVICE KERNEL FIRST TO LAST START [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "DISPATCH TOTAL CQ CMD OP TIME [ns]",
    "DISPATCH GO SEND WAIT TIME [ns]",
    "INPUTS",
    "OUTPUTS",
    "METAL TRACE ID",
    "METAL TRACE REPLAY SESSION ID",
    "COMPUTE KERNEL SOURCE",
    "COMPUTE KERNEL HASH",
    "DATA MOVEMENT KERNEL SOURCE",
    "DATA MOVEMENT KERNEL HASH",
    "BRISC MAX KERNEL SIZE [B]",
    "NCRISC MAX KERNEL SIZE [B]",
    "TRISC 0 MAX KERNEL SIZE [B]",
    "TRISC 1 MAX KERNEL SIZE [B]",
    "TRISC 2 MAX KERNEL SIZE [B]",
    "ERISC MAX KERNEL SIZE [B]",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM REQ I BW",
    "PM REQ O BW",
    "PM FPU UTIL (%)",
    "NOC UTIL (%)",
    "DRAM BW UTIL (%)",
    "NPE CONG IMPACT (%)",
]


def csv_header_format(header):
    return header.replace("_", " ").upper()


def import_tracy_op_logs(logFolder):
    logger.info(f"Importing ops logs")
    ops = {}
    signposts = {}
    signpostsCount = 0
    cached_ops = {}

    tracyOpTimesLog = os.path.join(logFolder, TRACY_OPS_TIMES_FILE_NAME)
    tracyOpDataLog = os.path.join(logFolder, TRACY_OPS_DATA_FILE_NAME)

    if not os.path.isfile(tracyOpTimesLog) or not os.path.isfile(tracyOpDataLog):
        return ops, signposts, None

    with open(tracyOpDataLog, "r", newline="") as csvFile:
        opDataDicts = csv.DictReader(csvFile, delimiter=";", quotechar="`")
        opsData = []
        traceIDs = {}
        traceReplays = {}
        for opDataDict in opDataDicts:
            opDataStr = opDataDict["MessageName"]
            opDataTime = opDataDict["total_ns"]
            if "TT_DNN" in opDataStr or "TT_METAL" in opDataStr:
                if "OP" in opDataStr:
                    tmpStrs = opDataStr.split(" ->\n", 1)
                    opData = {}
                    if len(tmpStrs) > 1:  # uncached device op, host op, or fallback op
                        jsonStr = tmpStrs[-1]
                        opData = json.loads(jsonStr)
                        opData["metal_trace_id"] = None
                        if "op_hash" in opData.keys():
                            assert "device_id" in opData.keys()
                            deviceID = int(opData["device_id"])
                            opHash = int(opData["op_hash"])
                            if deviceID in cached_ops.keys():
                                cached_ops[deviceID][opHash] = opData.copy()
                            else:
                                cached_ops[deviceID] = {opHash: opData.copy()}
                            del cached_ops[deviceID][opHash]["global_call_count"]
                            if deviceID in traceIDs:
                                opData["metal_trace_id"] = traceIDs[deviceID]
                    else:  # cached device op
                        opDataList = opDataStr.split(":", 1)[-1].split(",")
                        assert len(opDataList) > 3, "Wrong cached op info format"
                        opCode = opDataList[0].strip()
                        opHash = int(opDataList[1])
                        deviceID = int(opDataList[2])
                        opID = int(opDataList[3])
                        assert deviceID in cached_ops.keys(), "Expected hashed op info is not found"
                        assert opHash in cached_ops[deviceID].keys(), "Expected hashed op info is not found"
                        opData = cached_ops[deviceID][opHash].copy()
                        opData["global_call_count"] = opID
                        opData["metal_trace_id"] = None
                        if deviceID in traceIDs:
                            opData["metal_trace_id"] = traceIDs[deviceID]
                    opData["tracy_time"] = opDataTime
                    opsData.append(opData)
                elif "TRACE" in opDataStr:
                    IDs = opDataStr.split(":")[-1].strip().split(",")
                    assert len(IDs) == 2, (
                        "Wrong number of IDs is provided in trace message. "
                        "Device and trace are the two IDs that should be provided. "
                        f"But IDs {IDs} were provided"
                    )
                    deviceID = int(IDs[0].strip())
                    traceID = int(IDs[1].strip())
                    if "BEGIN" in opDataStr:
                        traceIDs[deviceID] = traceID
                    elif "END" in opDataStr:
                        assert traceIDs[deviceID] == traceID, (
                            f"Wrong trace ID, device {deviceID} should finish on trace ID "
                            f"{traceIDs[deviceID]} but it is finishing on trace ID {traceID}"
                        )
                        traceIDs[deviceID] = None
                    elif "REPLAY" in opDataStr:
                        replayIDTime = opDataTime

                        if deviceID in traceReplays:
                            if traceID in traceReplays[deviceID]:
                                traceReplays[deviceID][traceID].append(replayIDTime)
                            else:
                                traceReplays[deviceID][traceID] = [replayIDTime]
                        else:
                            traceReplays[deviceID] = {traceID: [replayIDTime]}

            if "TT_SIGNPOST" in opDataStr:
                signpostsCount += 1
                signposts[f"sp_{signpostsCount}"] = {"data": opDataStr, "tracy_time": opDataTime}
    for opData in opsData:
        ops[opData["global_call_count"]] = opData

    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for op in csvReader:
            if "TT_DNN" in op["name"] or "TT_METAL" in op["name"]:
                opID = int(op["zone_text"].split(":")[-1])
                assert opID in ops.keys(), f"Op time for op {opID} must present"
                ops[opID]["host_time"] = op

    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for op in csvReader:
            if op["special_parent_text"] and "id:" in op["special_parent_text"]:
                parentOpID = int(op["special_parent_text"].split(":")[-1])

                if "child_calls" in ops[parentOpID].keys():
                    if op["name"] in ops[parentOpID]["child_calls"].keys():
                        ops[parentOpID]["child_calls"][op["name"]] += int(op["exec_time_ns"])
                    else:
                        ops[parentOpID]["child_calls"][op["name"]] = int(op["exec_time_ns"])
                else:
                    ops[parentOpID]["child_calls"] = {op["name"]: int(op["exec_time_ns"])}

    return ops, signposts, traceReplays


def host_device_op_compare(op):
    if "metal_trace_replay_session_id" in op:
        return int(op["global_call_count"]), int(op["metal_trace_replay_session_id"])
    else:
        return int(op["global_call_count"]), 0


def device_op_compare_time(op):
    if "timeseries" in op and len(op["timeseries"]) > 0 and len(op["timeseries"][0]) > 1:
        return int(op["timeseries"][0][1])
    else:
        return 0


def device_op_compare_opID_time(op):
    if (
        "timeseries" in op
        and len(op["timeseries"]) > 0
        and len(op["timeseries"][0]) > 1
        and "run_host_id" in op["timeseries"][0][0]
    ):
        return int(op["timeseries"][0][0]["run_host_id"]), int(op["timeseries"][0][1])
    elif "timeseries" in op and len(op["timeseries"]) > 0 and len(op["timeseries"][0]) > 1:
        return 0, int(op["timeseries"][0][1])
    else:
        return 0, 0


# Generate a map of OP reference list per device.
def get_device_op_data(ops):
    logger.info(f"Getting device ops")
    deviceOps = {}
    hasTraceRuns = False
    for opID, opData in ops.items():
        if "device_id" in opData.keys():
            deviceID = opData["device_id"]
            if deviceID not in deviceOps.keys():
                deviceOps[deviceID] = [opData]
            else:
                deviceOps[deviceID].append(opData)
        if "metal_trace_id" in opData.keys() and opData["metal_trace_id"] is not None:
            hasTraceRuns = True

    for deviceID in deviceOps:
        deviceOps[deviceID].sort(key=host_device_op_compare)

    return deviceOps, hasTraceRuns


def extract_dispatch_op_id(dispatchOps):
    opId = 0
    for ts in dispatchOps["timeseries"]:
        if "meta_data" in ts[0] and "workers_runtime_id" in ts[0]["meta_data"]:
            metaData = eval(ts[0]["meta_data"])
            opId = metaData["workers_runtime_id"]
            break
    return opId


# Append device data to device ops and return the list of mapped device op ref list
def append_device_data(ops, traceReplays, logFolder, analyze_noc_traces, device_analysis_types):
    traceReplayCounts = {}
    for deviceID in traceReplays:
        traceReplayCounts[deviceID] = {}
        for traceID in traceReplays[deviceID]:
            traceReplayCounts[deviceID][traceID] = len(traceReplays[deviceID][traceID])
    devicesOps, hasTraceRuns = get_device_op_data(ops)
    logger.info(f"Appending device data")
    deviceTimesLog = os.path.join(logFolder, PROFILER_DEVICE_SIDE_LOG)
    traceOps = {}
    if os.path.isfile(deviceTimesLog):
        setup = device_post_proc_config.default_setup()
        if device_analysis_types:
            allAnalysis = setup.timerAnalysis
            pickedAnalysis = {}
            for analysis in device_analysis_types:
                assert analysis in allAnalysis.keys(), f" {analysis} is not calculated in device analysis"
                pickedAnalysis[analysis] = allAnalysis[analysis]

            setup.timerAnalysis = pickedAnalysis
        setup.deviceInputLog = deviceTimesLog
        deviceData = import_log_run_stats(setup)
        freq = deviceData["deviceInfo"]["freq"]
        arch = deviceData["deviceInfo"]["arch"]  # passed to NPE later
        for device in devicesOps:
            assert device in deviceData["devices"].keys()
            deviceOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
            deviceDispatchOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["dispatch_ops"]
            deviceOpsTime.sort(key=device_op_compare_time)
            if hasTraceRuns:
                generatedHostData = []
                opIDHostDataDict = {}
                for deviceOp in devicesOps[device]:
                    opID = deviceOp["global_call_count"]
                    assert (
                        opID not in opIDHostDataDict
                    ), f"Host op ID cannot be repeated: op ID {opID} was reported twice by the host"
                    opIDHostDataDict[opID] = copy.deepcopy(deviceOp)

                traceOps = {}
                for deviceOpTime in deviceOpsTime:
                    if len(deviceOpTime["timeseries"]) > 0:
                        timeID, ts, statData, risc, core = deviceOpTime["timeseries"][0]
                        assert "run_host_id" in timeID.keys(), "Device op ID missing: Device data must provide op ID"
                        deviceOpID = timeID["run_host_id"]
                        assert (
                            deviceOpID in opIDHostDataDict
                        ), f"Device op ID not present: Device op ID {deviceOpID} not present in host data on device {device}"
                        traceID = opIDHostDataDict[deviceOpID]["metal_trace_id"]
                        if traceID is not None:
                            if device in traceOps:
                                if traceID in traceOps[device]:
                                    if deviceOpID in traceOps[device][traceID]:
                                        traceReplays[device][traceID].pop(0)
                                        traceOps[device][traceID] = set([deviceOpID])
                                    else:
                                        traceOps[device][traceID].add(deviceOpID)
                                else:
                                    traceOps[device][traceID] = set([deviceOpID])
                            else:
                                traceOps[device] = {traceID: set([deviceOpID])}
                            assert (
                                len(traceReplays[device][traceID]) > 0
                            ), "Wrong trace replay count: Device has more ops than trace replay issued commands"
                            opIDHostDataDict[deviceOpID]["tracy_time"] = traceReplays[device][traceID][0]
                            opIDHostDataDict[deviceOpID]["metal_trace_replay_session_id"] = (
                                traceReplayCounts[device][traceID] - len(traceReplays[device][traceID]) + 1
                            )
                        generatedHostData.append(copy.deepcopy(opIDHostDataDict[deviceOpID]))
                devicesOps[device] = generatedHostData

            deviceOpsTime.sort(key=device_op_compare_opID_time)
            devicesOps[device].sort(key=host_device_op_compare)

            dispatchOPAnalysis = {}
            for deviceDispatchOp in deviceDispatchOpsTime:
                dispatchOpID = extract_dispatch_op_id(deviceDispatchOp)
                dispatchOPAnalysis[dispatchOpID] = deviceDispatchOp["analysis"]

            # attach op dispatch analysis to op analysis
            for deviceOp in deviceOpsTime:
                opID = deviceOp["timeseries"][0][0]["run_host_id"]
                if opID in dispatchOPAnalysis:
                    for dispatchAnalysis in dispatchOPAnalysis[opID]:
                        deviceOp["analysis"][dispatchAnalysis] = dispatchOPAnalysis[opID][dispatchAnalysis]
                    del dispatchOPAnalysis[opID]

            assert len(dispatchOPAnalysis) == 0, "Unrecognized dispatch OPs are presentent by dispatch cores"

            if len(devicesOps[device]) != len(deviceOpsTime):
                deviceOPId = None
                hostOPId = None
                for deviceOp, deviceOpTime in zip(devicesOps[device], deviceOpsTime):
                    if len(deviceOpTime["timeseries"]) > 0:
                        timeID, ts, statData, risc, core = deviceOpTime["timeseries"][0]
                        if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                            if "run_host_id" in timeID.keys():
                                if timeID["run_host_id"] != deviceOp["global_call_count"]:
                                    deviceOPId = timeID["run_host_id"]
                                    hostOPId = deviceOp["global_call_count"]
                                    break

                if deviceOPId and hostOPId:
                    assert False, (
                        f"Device data mismatch: Expected {len(devicesOps[device])} "
                        f"but received {len(deviceOpsTime)} ops on device {device}. "
                        f"Device is showing op ID {deviceOPId} when host is showing op ID {hostOPId}"
                    )
                else:
                    assert (
                        False
                    ), f"Device data mismatch: Expected {len(devicesOps[device])} but received {len(deviceOpsTime)} ops on device {device}"
            for deviceOp, deviceOpTime in zip(devicesOps[device], deviceOpsTime):
                cores = set()
                for timeID, ts, statData, risc, core in deviceOpTime["timeseries"]:
                    if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                        if "run_host_id" in timeID.keys():
                            assert (
                                timeID["run_host_id"] == deviceOp["global_call_count"]
                            ), f"op id {timeID['run_host_id']} reported by device {device} is not matching assigned op id {deviceOp['global_call_count']}"
                        if core not in cores:
                            cores.add(core)
                deviceOp["core_usage"] = {"count": len(cores), "cores": [str(core) for core in cores]}
                deviceOp["device_time"] = {
                    analysis: {"series": data["series"], "stats": data["stats"]}
                    for analysis, data in deviceOpTime["analysis"].items()
                }
                for analysis, data in deviceOp["device_time"].items():
                    for sample in data["series"]:
                        sample["duration_ns"] = sample["duration_cycles"] * 1000 / freq
            traceOps = {}

            # Tag trace ops with a UID
            for device in devicesOps:
                for deviceOp in devicesOps[device]:
                    if "metal_trace_replay_session_id" in deviceOp.keys():
                        deviceOp["global_call_count"] = (
                            deviceOp["global_call_count"]
                            | deviceOp["metal_trace_replay_session_id"] << TRACE_OP_ID_BITSHIFT
                        )
                        traceOps[deviceOp["global_call_count"]] = deviceOp
                    else:
                        # Update host reported device op with device populated version
                        ops[deviceOp["global_call_count"]] = deviceOp

    # if enabled, analyze noc trace files present in log folder and add
    # relevant statistics to 'ops' dict
    if analyze_noc_traces:
        npe_stats = analyzeNoCTraces(logFolder, arch)
        if npe_stats is not None:
            ops_found = 0
            for op_id in ops:
                op_npe_stats = npe_stats.getDatapointByID(op_id)
                if op_npe_stats is not None:
                    ops_found += 1
                    ops[op_id]["NOC UTIL (%)"] = round(op_npe_stats.result.overall_avg_link_util, 1)
                    ops[op_id]["DRAM BW UTIL (%)"] = round(op_npe_stats.result.dram_bw_util, 1)
                    ops[op_id]["NPE CONG IMPACT (%)"] = round(op_npe_stats.result.getCongestionImpact(), 2)
            logger.info(f"Analyzed {ops_found} operations with tt-npe trace data.")

    return devicesOps, traceOps


def get_device_data_generate_report(
    logFolder, outputFolder, date, nameAppend, export_csv=True, cleanup_device_log=False, device_analysis_types=[]
):
    deviceTimesLog = os.path.join(logFolder, PROFILER_DEVICE_SIDE_LOG)
    devicePreOpTime = {}
    devicePreOpDMStartTime = {}
    deviceOps = {}
    i = 0
    rowDicts = []
    perCoreRowDicts = []
    perCoreCSVHeader = set()

    outFolder = PROFILER_OUTPUT_DIR
    if outputFolder:
        outFolder = outputFolder

    name = OUT_NAME
    perCoreName = PER_CORE_OP_TO_OP_OUT_NAME
    outFolder = os.path.abspath(outFolder)

    if nameAppend:
        name += f"_{nameAppend}"
        perCoreName += f"_{nameAppend}"
        outFolder = os.path.join(outFolder, nameAppend)

    if date:
        dateStr = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        name += f"_{dateStr}"
        perCoreName += f"_{dateStr}"
        outFolder = os.path.join(outFolder, dateStr)

    if export_csv:
        allOpsCSVPath = os.path.join(outFolder, f"{name}.csv")
        perCoreCSVPath = os.path.join(outFolder, f"{perCoreName}.csv")
        logger.info(f"Copying runtime artifacts")
        os.system(f"rm -rf {outFolder}; mkdir -p {outFolder}")
        if os.path.isfile(f"{logFolder / PROFILER_DEVICE_SIDE_LOG}"):
            os.system(f"cp {logFolder / PROFILER_DEVICE_SIDE_LOG} {outFolder}")

    if os.path.isfile(deviceTimesLog):
        logger.info(f"Getting device only ops data")
        setup = device_post_proc_config.default_setup()
        if device_analysis_types:
            allAnalysis = setup.timerAnalysis
            pickedAnalysis = {}
            for analysis in device_analysis_types:
                assert analysis in allAnalysis.keys(), f" {analysis} is not calculated in device analysis"
                pickedAnalysis[analysis] = allAnalysis[analysis]

            setup.timerAnalysis = pickedAnalysis
        setup.deviceInputLog = deviceTimesLog
        deviceData = import_log_run_stats(setup)
        logger.info(f"Generating device op report ...")
        freq = deviceData["deviceInfo"]["freq"]
        for device in deviceData["devices"]:
            deviceOps[device] = []
            deviceOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
            for deviceOpTime in deviceOpsTime:
                i += 1
                deviceOp = {}
                cores = set()
                for timeID, ts, statData, risc, core in deviceOpTime["timeseries"]:
                    if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                        if core not in cores:
                            cores.add(core)
                deviceOp["core_usage"] = {"count": len(cores), "cores": [str(core) for core in cores]}
                deviceOp["device_time"] = {
                    analysis: {"series": data["series"], "stats": data["stats"]}
                    for analysis, data in deviceOpTime["analysis"].items()
                }

                if "run_host_id" in timeID.keys():
                    deviceOp["global_call_count"] = timeID["run_host_id"]
                else:
                    deviceOp["global_call_count"] = i
                for analysis, data in deviceOp["device_time"].items():
                    for sample in data["series"]:
                        sample["duration_ns"] = sample["duration_cycles"] * 1000 / freq
                deviceOps[device].append(deviceOp)

                rowDict = {csv_header_format("global_call_count"): deviceOp["global_call_count"]}
                for analysis, data in deviceOp["device_time"].items():
                    analysisData = data["series"]
                    analysisStats = data["stats"]
                    if "per_core" in analysis:
                        assert len(analysisData) >= 1, "Unexpected device data format"
                        headerField = f"{csv_header_format(analysis)} MIN [ns]"
                        rowDict[headerField] = f"{analysisStats['Min'] * 1000 / freq:.0f}"
                        headerField = f"{csv_header_format(analysis)} MAX [ns]"
                        rowDict[headerField] = f"{analysisStats['Max'] * 1000 / freq:.0f}"
                        headerField = f"{csv_header_format(analysis)} AVG [ns]"
                        rowDict[headerField] = f"{analysisStats['Average'] * 1000 / freq:.0f}"
                    else:
                        headerField = f"{csv_header_format(analysis)} [ns]"
                        assert len(analysisData) == 1, "Unexpected device data format"
                        rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                    if analysis == "device_fw_duration":
                        rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                        rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]
                    if analysis == "device_kernel_duration":
                        if device in devicePreOpTime.keys():
                            rowDict["OP TO OP LATENCY [ns]"] = round(
                                1000 * (analysisData[0]["start_cycle"] - devicePreOpTime[device]) / freq
                            )
                        else:
                            rowDict["OP TO OP LATENCY [ns]"] = 0
                        devicePreOpTime[device] = analysisData[0]["end_cycle"]
                    if analysis == "device_kernel_duration_dm_start":
                        if device in devicePreOpDMStartTime.keys():
                            rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = round(
                                1000 * (analysisData[0]["start_cycle"] - devicePreOpDMStartTime[device]) / freq
                            )
                        else:
                            rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = 0
                        devicePreOpDMStartTime[device] = analysisData[0]["end_cycle"]
                rowDicts.append(rowDict)

            def get_core_str_format(core):
                return f"{core[0]}; {core[1]} [ns]"

            allCores = list(deviceData["devices"][device]["cores"].keys())
            allCores.remove("DEVICE")
            allCores.sort()
            for core in allCores:
                perCoreCSVHeader.add(get_core_str_format(core))

            coreOpToOps = {}
            opToOps = []
            for core in allCores:
                deviceDataDict = deviceData["devices"][device]["cores"][core]["riscs"]["TENSIX"]
                if "analysis" in deviceDataDict:
                    coreSeries = deviceDataDict["analysis"]["op2op"]["series"]
                    for op2op in coreSeries:
                        if op2op["end_iter_mark"][1] != op2op["start_iter_mark"][1]:
                            startMarker, endMarker = op2op["duration_type"]
                            op2opID = (startMarker["run_host_id"], endMarker["run_host_id"])
                            op2opDuration = op2op["duration_cycles"]
                            op2opStart = op2op["start_cycle"]
                            opToOps.append((op2opStart, op2opID, op2opDuration, core))
                            if core in coreOpToOps:
                                coreOpToOps[core].append((op2opStart, op2opID, op2opDuration, core))
                            else:
                                coreOpToOps[core] = deque([(op2opStart, op2opID, op2opDuration, core)])
            opToOps.sort()

            pickedOps = set()
            for op2op in opToOps:
                if op2op not in pickedOps:
                    op2opStart, op2opID, op2opDuration, core = op2op
                    perCoreRowDict = {
                        "device ID": device,
                        "op2op ID": f"{op2opID[0]} -> {op2opID[1]}",
                    }
                    for core, series in coreOpToOps.items():
                        perCoreRowDict[get_core_str_format(core)] = ""
                        if series and op2opID == series[0][1]:
                            coreOpToOp = series.popleft()
                            perCoreRowDict[get_core_str_format(core)] = (
                                coreOpToOp[2] - PROFILER_OP_TO_OP_OVERHEAD_NANO_SEC
                            )
                            pickedOps.add(coreOpToOp)

                    perCoreRowDicts.append(perCoreRowDict)

        rowDictHeaders = set()
        for row in rowDicts:
            for k in row.keys():
                rowDictHeaders.add(k)
        if export_csv:
            with open(allOpsCSVPath, "w") as allOpsCSV:
                allHeaders = []
                for header in OPS_CSV_HEADER:
                    if header in rowDictHeaders:
                        allHeaders.append(header)
                writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
                writer.writeheader()
                for rowDict in rowDicts:
                    for field, fieldData in rowDict.items():
                        rowDict[field] = str(fieldData).replace(",", ";")
                    writer.writerow(rowDict)
            logger.info(f"Device only OPs csv generated at: {allOpsCSVPath}")
            with open(perCoreCSVPath, "w") as perCoreCSV:
                perCoreCSVHeader = ["device ID", "op2op ID"] + [core for core in perCoreCSVHeader]

                writer = csv.DictWriter(perCoreCSV, fieldnames=perCoreCSVHeader)
                writer.writeheader()

                for rowDict in perCoreRowDicts:
                    writer.writerow(rowDict)
            logger.info(f"Device only per core op to op times csv generated at: {perCoreCSVPath}")

        if cleanup_device_log:
            os.remove(deviceTimesLog)
    else:
        logger.info("No device logs found")
    return rowDicts


def generate_reports(ops, deviceOps, traceOps, signposts, logFolder, outputFolder, date, nameAppend):
    logger.info(f"OPs' perf analysis is finished! Generating reports ...")
    outFolder = PROFILER_OUTPUT_DIR
    if outputFolder:
        outFolder = outputFolder

    name = OUT_NAME
    outFolder = os.path.abspath(outFolder)

    if nameAppend:
        name += f"_{nameAppend}"
        outFolder = os.path.join(outFolder, nameAppend)

    if date:
        dateStr = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        name += f"_{dateStr}"
        outFolder = os.path.join(outFolder, dateStr)
    allOpsCSVPath = os.path.join(outFolder, f"{name}.csv")

    logger.info(f"Copying runtime artifacts")
    os.system(f"rm -rf {outFolder}; mkdir -p {outFolder}")
    if os.path.isfile(f"{logFolder / TRACY_FILE_NAME}"):
        os.system(f"cp {logFolder / TRACY_FILE_NAME} {outFolder}")
    if os.path.isfile(f"{logFolder / PROFILER_DEVICE_SIDE_LOG}"):
        os.system(f"cp {logFolder / PROFILER_DEVICE_SIDE_LOG} {outFolder}")
    if os.path.isdir(f"{logFolder.parent / 'npe_viz'}"):
        os.system(f"cp -r {logFolder.parent / 'npe_viz'} {outFolder}")

    logger.info(f"Generating OPs CSV")
    allOpsCSVPath = os.path.join(outFolder, f"{name}.csv")
    with open(allOpsCSVPath, "w") as allOpsCSV:
        rowDicts = []

        devicePreOpTime = {}
        devicePreOpDMStartTime = {}

        tensorCSVData = {
            "INPUT": {
                "maxCount": -1,
                "headers": [],
            },
            "OUTPUT": {
                "maxCount": -1,
                "headers": [],
            },
        }

        def io_tensor_to_csv(ioField, ioData):
            headers = []
            data = {}
            if ioField == "shape":
                for field in ["W", "Z", "Y", "X"]:
                    headers.append(field)
                    assert field in ioData.keys(), "Wrong io tensor shape data format"
                    data[field] = ioData[field]
            elif ioField == "dtype":
                headers = ["DATATYPE"]
                data["DATATYPE"] = ioData
            elif ioField == "layout":
                headers = ["LAYOUT"]
                data["LAYOUT"] = ioData
            elif ioField == "storage_type":
                headers = ["MEMORY"]
                if type(ioData) == str:
                    data["MEMORY"] = ioData
                else:
                    assert "device_id" in ioData.keys(), "Wrong io tensor memory data format"
                    deviceID = ioData["device_id"]
                    assert "memory_config" in ioData.keys(), "Wrong io tensor memory data format"
                    assert "buffer_type" in ioData["memory_config"].keys(), "Wrong io tensor memory data format"
                    bufferType = ioData["memory_config"]["buffer_type"].upper()
                    assert "memory_layout" in ioData["memory_config"].keys(), "Wrong io tensor memory data format"
                    memoryLayout = ioData["memory_config"]["memory_layout"].upper()
                    data["MEMORY"] = f"DEV_{deviceID}_{bufferType}_{memoryLayout}"

            return headers, data

        def add_io_data(tensors, ioType):
            ioFields = ["shape", "layout", "dtype", "storage_type"]
            for count, tensor in enumerate(tensors):
                for ioField in ioFields:
                    assert ioField in tensor.keys(), "Wrong io tensor fields"
                    ioData = tensor[ioField]
                    fields, data = io_tensor_to_csv(ioField, ioData)
                    for field in fields:
                        header = f"{ioType}_{count}_{field}".upper()
                        rowDict[header] = data[field]
                        if count > tensorCSVData[ioType]["maxCount"]:
                            tensorCSVData[ioType]["headers"].append(header)
                if count > tensorCSVData[ioType]["maxCount"]:
                    tensorCSVData[ioType]["maxCount"] = count

        def row_compare(row):
            ret = 0
            if type(row) is str and "sp" in row:
                ret = signposts[row]["tracy_time"]
            elif type(row) is int:
                if row > ((1 << TRACE_OP_ID_BITSHIFT) - 1):
                    ret = traceOps[row]["tracy_time"]
                else:
                    ret = ops[row]["host_time"]["ns_since_start"]
            ret = int(ret)
            return ret

        rowKeys = list(ops.keys()) + list(traceOps.keys()) + list(signposts.keys())
        rowKeys.sort(key=row_compare)
        childCallKeys = set()
        for row in rowKeys:
            if type(row) is int:
                if row > ((1 << TRACE_OP_ID_BITSHIFT) - 1):
                    opData = traceOps[row]
                else:
                    opData = ops[row]
                if "child_calls" in opData.keys():
                    for childCall in opData["child_calls"]:
                        childCallKeys.add(f"{childCall}_TT_HOST_FUNC [ns]")

        for row in rowKeys:
            rowDict = {}
            if type(row) is str and "sp" in row:
                headerAndMessage = signposts[row]["data"].split(": ")[-1].split("\n")
                rowDict["OP CODE"] = headerAndMessage[0]
                rowDict["OP TYPE"] = "signpost"
                if len(headerAndMessage) > 1:
                    rowDict["ATTRIBUTES"] = headerAndMessage[1]
                rowDict["HOST START TS"] = int(signposts[row]["tracy_time"])
            elif type(row) is int:
                op = row
                if op > ((1 << TRACE_OP_ID_BITSHIFT) - 1):
                    opData = traceOps[op]
                    opData["global_call_count"] = ((1 << TRACE_OP_ID_BITSHIFT) - 1) & op
                else:
                    opData = ops[op]
                    opData["metal_trace_replay_session_id"] = ""
                    if "trac_id" not in opData.keys() or opData["metal_trace_id"] is None:
                        opData["metal_trace_id"] = ""

                for field, fieldData in opData.items():
                    headerField = csv_header_format(field)
                    if headerField in OPS_CSV_HEADER:
                        rowDict[headerField] = fieldData

                assert "host_time" in opData.keys(), "Corrupted op data"
                rowDict["HOST START TS"] = int(opData["host_time"]["ns_since_start"])
                rowDict["HOST END TS"] = int(opData["host_time"]["ns_since_start"]) + int(
                    opData["host_time"]["exec_time_ns"]
                )
                rowDict["HOST DURATION [ns]"] = int(opData["host_time"]["exec_time_ns"])

                if "NOC UTIL (%)" in opData:
                    rowDict["NOC UTIL (%)"] = opData.get("NOC UTIL (%)")
                if "DRAM BW UTIL (%)" in opData:
                    rowDict["DRAM BW UTIL (%)"] = opData.get("DRAM BW UTIL (%)")
                if "NPE CONG IMPACT (%)" in opData:
                    rowDict["NPE CONG IMPACT (%)"] = opData.get("NPE CONG IMPACT (%)")

                if "kernel_info" in opData.keys():
                    rowDict["COMPUTE KERNEL SOURCE"] = []
                    rowDict["COMPUTE KERNEL HASH"] = []
                    rowDict["DATA MOVEMENT KERNEL SOURCE"] = []
                    rowDict["DATA MOVEMENT KERNEL HASH"] = []
                    for computeKernel in opData["kernel_info"]["compute_kernels"]:
                        rowDict["MATH FIDELITY"] = computeKernel["math_fidelity"]
                        rowDict["COMPUTE KERNEL SOURCE"].append(computeKernel["source"])
                        rowDict["COMPUTE KERNEL HASH"].append(computeKernel["name"])

                    for dmKernel in opData["kernel_info"]["datamovement_kernels"]:
                        rowDict["DATA MOVEMENT KERNEL SOURCE"].append(dmKernel["source"])
                        rowDict["DATA MOVEMENT KERNEL HASH"].append(dmKernel["name"])

                    for kernel, kernelSize in opData["kernel_info"]["kernel_sizes"].items():
                        rowDict[kernel.upper().replace("_", " ") + " [B]"] = kernelSize

                if "core_usage" in opData.keys():
                    rowDict["CORE COUNT"] = opData["core_usage"]["count"]

                if "device_time" in opData.keys():
                    assert "device_id" in opData.keys(), "Op has device data without device_id"
                    deviceID = opData["device_id"]
                    for analysis, data in opData["device_time"].items():
                        analysisData = data["series"]
                        analysisStats = data["stats"]
                        freq = analysisData[0]["duration_cycles"] / analysisData[0]["duration_ns"]
                        if "per_core" in analysis:
                            assert len(analysisData) >= 1, "Unexpected device data format"
                            headerField = f"{csv_header_format(analysis)} MIN [ns]"
                            rowDict[headerField] = f"{analysisStats['Min'] / freq:.0f}"
                            headerField = f"{csv_header_format(analysis)} MAX [ns]"
                            rowDict[headerField] = f"{analysisStats['Max'] / freq:.0f}"
                            headerField = f"{csv_header_format(analysis)} AVG [ns]"
                            rowDict[headerField] = f"{analysisStats['Average'] / freq:.0f}"
                        else:
                            headerField = f"{csv_header_format(analysis)} [ns]"
                            assert len(analysisData) == 1, "Unexpected device data format"
                            rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                        if analysis == "device_fw_duration":
                            rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                            rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]
                        if analysis == "device_kernel_duration":
                            if deviceID in devicePreOpTime.keys():
                                rowDict["OP TO OP LATENCY [ns]"] = round(
                                    (analysisData[0]["start_cycle"] - devicePreOpTime[deviceID]) / freq
                                )
                            else:
                                rowDict["OP TO OP LATENCY [ns]"] = 0
                            devicePreOpTime[deviceID] = analysisData[0]["end_cycle"]
                        if analysis == "device_kernel_duration_dm_start":
                            if deviceID in devicePreOpDMStartTime.keys():
                                rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = round(
                                    (analysisData[0]["start_cycle"] - devicePreOpDMStartTime[deviceID]) / freq
                                )
                            else:
                                rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = 0
                            devicePreOpDMStartTime[deviceID] = analysisData[0]["end_cycle"]

                if "child_calls" in opData.keys():
                    for childCall, duration in opData["child_calls"].items():
                        headerField = f"{childCall}_TT_HOST_FUNC [ns]"
                        rowDict[headerField] = f"{duration:.0f}"

                assert "input_tensors" in opData.keys(), "Ops must have input tensors"
                if "optional_input_tensors" in opData.keys():
                    add_io_data(opData["input_tensors"] + opData["optional_input_tensors"], "INPUT")
                else:
                    add_io_data(opData["input_tensors"], "INPUT")

                if "output_tensors" in opData.keys():
                    add_io_data(opData["output_tensors"], "OUTPUT")

                if "performance_model" in opData.keys():
                    rowDict["PM IDEAL [ns]"] = opData["performance_model"]["ideal_ns"]
                    rowDict["PM COMPUTE [ns]"] = opData["performance_model"]["compute_ns"]
                    rowDict["PM BANDWIDTH [ns]"] = opData["performance_model"]["bandwidth_ns"]
                    rowDict["PM REQ I BW"] = opData["performance_model"]["input_bws"]
                    rowDict["PM REQ O BW"] = opData["performance_model"]["output_bws"]

                    if "DEVICE KERNEL DURATION [ns]" in rowDict:
                        try:
                            fpu_util = (
                                100.0
                                * float(rowDict["PM COMPUTE [ns]"])
                                / float(rowDict["DEVICE KERNEL DURATION [ns]"])
                            )
                            rowDict["PM FPU UTIL (%)"] = round(fpu_util, 3)
                        except ZeroDivisionError:
                            rowDict["PM FPU UTIL (%)"] = 0.0

            rowDicts.append(rowDict)

        ioHeaderIndex = OPS_CSV_HEADER.index("INPUTS")
        allHeaders = (
            OPS_CSV_HEADER[:ioHeaderIndex]
            + tensorCSVData["INPUT"]["headers"]
            + tensorCSVData["OUTPUT"]["headers"]
            + OPS_CSV_HEADER[ioHeaderIndex + 2 :]
            + sorted(list(childCallKeys))
        )
        writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
        writer.writeheader()
        for rowDict in rowDicts:
            for field, fieldData in rowDict.items():
                rowDict[field] = str(fieldData).replace(",", ";")
            writer.writerow(rowDict)
    logger.info(f"OPs csv generated at: {allOpsCSVPath}")


def analyzeNoCTraces(logFolder, arch):
    """Attempts to import tt-npe from $PYTHONPATH and process noc traces to
    obtain per-operation DRAM BW and NoC utilization statistics and create
    visualizer timeline files"""
    try:
        from npe_analyze_noc_trace_dir import analyze_noc_traces_in_dir

        logger.info(f"tt-npe module imported successfully; analyzing noc traces ... ")
        return analyze_noc_traces_in_dir(
            noc_trace_dir=logFolder, device_name=arch, emit_viz_timeline_files=True, quiet=True
        )
    except ImportError:
        logger.warning("Could not import tt-npe module. Ensure tt-npe is built, then source 'tt-npe/ENV_SETUP'")
        return None
    except Exception as e:
        logger.error("Unexpected error occured when analyzing noc traces, aborting ... ")
        logger.error(" ↳ " + repr(e))
        return None


def process_ops(
    output_folder, name_append, date, device_only=False, analyze_noc_traces=False, device_analysis_types=[]
):
    if not output_folder:
        output_folder = PROFILER_ARTIFACTS_DIR
    logFolder = generate_logs_folder(output_folder)
    reportFolder = generate_reports_folder(output_folder)

    ops, signposts, traceReplays = import_tracy_op_logs(logFolder)

    if ops and not device_only:
        deviceOps, traceOps = append_device_data(
            ops, traceReplays, logFolder, analyze_noc_traces, device_analysis_types
        )
        generate_reports(ops, deviceOps, traceOps, signposts, logFolder, reportFolder, date, name_append)
    else:
        deviceOps = get_device_data_generate_report(
            logFolder, reportFolder, date, name_append, device_analysis_types=device_analysis_types
        )


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
@click.option("--device-only", default=False, is_flag=True, help="Only generate a device data report")
@click.option(
    "--analyze-noc-traces", is_flag=True, help="Use tt-npe to analyze profiler noc event trace files (if available)"
)
@click.option("-a", "--device-analysis-types", multiple=True, help="Subset of analysis types to be performed on device")
def main(output_folder, name_append, date, device_only, analyze_noc_traces, device_analysis_types):
    if output_folder:
        output_folder = Path(output_folder)
    process_ops(output_folder, name_append, date, device_only, analyze_noc_traces, device_analysis_types)


if __name__ == "__main__":
    main()
