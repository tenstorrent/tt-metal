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

import click
from loguru import logger

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import (
    PROFILER_LOGS_DIR,
    PROFILER_OPS_LOGS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_HOST_SIDE_LOG,
    PROFILER_OUTPUT_DIR,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
)

yaml.SafeDumper.ignore_aliases = lambda *args: True

OUT_NAME = "ops_perf_results"

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
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "INPUTS",
    "OUTPUTS",
    "COMPUTE KERNEL PATH",
    "COMPUTE KERNEL HASH",
    "DATA MOVEMENT KERNEL PATH",
    "DATA MOVEMENT KERNEL HASH",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM REQ I BW",
    "PM REQ O BW",
]


def csv_header_format(header):
    return header.replace("_", " ").upper()


def import_tracy_op_logs():
    logger.info(f"Importing ops logs")
    ops = {}
    signposts = {}
    signpostsCount = 0
    cached_ops = {}

    tracyOpTimesLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_TIMES_FILE_NAME)
    tracyOpDataLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_DATA_FILE_NAME)

    if not os.path.isfile(tracyOpTimesLog) or not os.path.isfile(tracyOpDataLog):
        return ops, signposts

    with open(tracyOpDataLog, "r", newline="") as csvFile:
        opDataDicts = csv.DictReader(csvFile, delimiter=";", quotechar="`")
        opsData = []
        for opDataDict in opDataDicts:
            opDataStr = opDataDict["MessageName"]
            opDataTime = opDataDict["total_ns"]
            if "TT_DNN" in opDataStr:
                tmpStrs = opDataStr.split(" ->\n", 1)
                opData = {}
                if len(tmpStrs) > 1:  # uncached device op, host op, or fallback op
                    jsonStr = tmpStrs[-1]
                    opData = json.loads(jsonStr)
                    if "op_hash" in opData.keys():
                        assert "device_id" in opData.keys()
                        deviceID = int(opData["device_id"])
                        opHash = int(opData["op_hash"])
                        if deviceID in cached_ops.keys():
                            cached_ops[deviceID][opHash] = opData.copy()
                        else:
                            cached_ops[deviceID] = {opHash: opData.copy()}
                        del cached_ops[deviceID][opHash]["global_call_count"]
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
                opData["tracy_time"] = opDataTime
                opsData.append(opData)

            if "TT_SIGNPOST" in opDataStr:
                signpostsCount += 1
                signposts[f"sp_{signpostsCount}"] = {"data": opDataStr, "tracy_time": opDataTime}
    for opData in opsData:
        ops[opData["global_call_count"]] = opData

    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for op in csvReader:
            if "TT_DNN" in op["name"]:
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

    return ops, signposts


# Generate a map of OP reference list per device.
def get_device_op_data(ops):
    logger.info(f"Getting device ops")
    deviceOps = {}
    for opID, opData in ops.items():
        if "device_id" in opData.keys():
            deviceID = opData["device_id"]
            if deviceID not in deviceOps.keys():
                deviceOps[deviceID] = [opData]
            else:
                deviceOps[deviceID].append(opData)

    def device_ops_compare(op):
        return int(op["global_call_count"])

    for deviceID in deviceOps:
        deviceOps[deviceID].sort(key=device_ops_compare)

    return deviceOps


# Append device data to device ops and return the list of mapped device op ref list
def append_device_data(ops, deviceLogFolder):
    deviceOps = get_device_op_data(ops)
    logger.info(f"Appending device data")
    deviceTimesLog = os.path.join(deviceLogFolder, PROFILER_DEVICE_SIDE_LOG)
    if os.path.isfile(deviceTimesLog):
        setup = device_post_proc_config.default_setup()
        setup.deviceInputLog = deviceTimesLog
        deviceData = import_log_run_stats(setup)
        freq = deviceData["deviceInfo"]["freq"]
        for device in deviceOps:
            assert device in deviceData["devices"].keys()
            deviceOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
            if len(deviceOps[device]) != len(deviceOpsTime):
                deviceOPId = None
                hostOPId = None
                for deviceOp, deviceOpTime in zip(deviceOps[device], deviceOpsTime):
                    if len(deviceOpTime["timeseries"]) > 0:
                        timeID, ts, statData, risc, core = deviceOpTime["timeseries"][0]
                        if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                            if "run_host_id" in timeID.keys():
                                if timeID["run_host_id"] != deviceOp["global_call_count"]:
                                    deviceOPId = timeID["run_host_id"]
                                    hostOPId = deviceOp["global_call_count"]
                                    break

                if deviceOPId and hostOPId:
                    assert (
                        False
                    ), f"Device data mismatch: Expected {len(deviceOps[device])} but received {len(deviceOpsTime)} ops on device {device}. Device is showing op ID {deviceOPId} when host is showing op ID {hostOPId}"
                else:
                    assert (
                        True
                    ), f"Device data mismatch: Expected {len(deviceOps[device])} but received {len(deviceOpsTime)} ops on device {device}"
            for deviceOp, deviceOpTime in zip(deviceOps[device], deviceOpsTime):
                cores = set()
                for timeID, ts, statData, risc, core in deviceOpTime["timeseries"]:
                    if "zone_name" in timeID.keys() and "FW" in timeID["zone_name"]:
                        if "run_host_id" in timeID.keys():
                            assert (
                                timeID["run_host_id"] == deviceOp["global_call_count"]
                            ), f"op id {timeID['run_host_id']} reproted by device is not matching assigned op id {deviceOp['global_call_count']}"
                        if core not in cores:
                            cores.add(core)
                deviceOp["core_usage"] = {"count": len(cores), "cores": [str(core) for core in cores]}
                deviceOp["device_time"] = {
                    analysis: data["series"] for analysis, data in deviceOpTime["analysis"].items()
                }
                for analysis, data in deviceOp["device_time"].items():
                    for sample in data:
                        sample["duration_ns"] = sample["duration_cycles"] * 1000 / freq
    return deviceOps


def get_device_data_generate_report(deviceLogFolder, outputFolder, date, nameAppend):
    deviceTimesLog = os.path.join(deviceLogFolder, PROFILER_DEVICE_SIDE_LOG)
    devicePreOpTime = {}
    deviceOps = {}
    i = 0
    rowDicts = []

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
    if os.path.isfile(f"{PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG}"):
        os.system(f"cp {PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG} {outFolder}")

    if os.path.isfile(deviceTimesLog):
        logger.info(f"Getting device only ops data")
        setup = device_post_proc_config.default_setup()
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
                    analysis: data["series"] for analysis, data in deviceOpTime["analysis"].items()
                }

                if "run_host_id" in timeID.keys():
                    deviceOp["global_call_count"] = timeID["run_host_id"]
                else:
                    deviceOp["global_call_count"] = i
                for analysis, data in deviceOp["device_time"].items():
                    for sample in data:
                        sample["duration_ns"] = sample["duration_cycles"] * 1000 / freq
                deviceOps[device].append(deviceOp)

                rowDict = {csv_header_format("global_call_count"): deviceOp["global_call_count"]}
                for analysis, analysisData in deviceOp["device_time"].items():
                    headerField = f"{csv_header_format(analysis)} [ns]"
                    assert len(analysisData) == 1, "Unexpected device data format"
                    rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                    if analysis == "device_fw_duration":
                        rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                        rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]
                        if device in devicePreOpTime.keys():
                            rowDict["OP TO OP LATENCY [ns]"] = round(
                                1000 * (analysisData[0]["start_cycle"] - devicePreOpTime[device]) / freq
                            )
                        else:
                            rowDict["OP TO OP LATENCY [ns]"] = 0
                        devicePreOpTime[device] = analysisData[0]["end_cycle"]
                rowDicts.append(rowDict)

        with open(allOpsCSVPath, "w") as allOpsCSV:
            allHeaders = []
            for header in OPS_CSV_HEADER:
                if header in rowDicts[-1].keys():
                    allHeaders.append(header)
            writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
            writer.writeheader()
            for rowDict in rowDicts:
                for field, fieldData in rowDict.items():
                    rowDict[field] = str(fieldData).replace(",", ";")
                writer.writerow(rowDict)
        logger.info(f"Device only OPs csv generated at: {allOpsCSVPath}")
    else:
        logger.info("No device logs found")
    return deviceOps


def generate_reports(ops, deviceOps, signposts, outputFolder, date, nameAppend):
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
    if os.path.isfile(f"{PROFILER_LOGS_DIR / TRACY_FILE_NAME}"):
        os.system(f"cp {PROFILER_LOGS_DIR / TRACY_FILE_NAME} {outFolder}")
    if os.path.isfile(f"{PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG}"):
        os.system(f"cp {PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG} {outFolder}")

    # logger.info(f"Generating OPs yaml")
    # allOpsYAMLPath = os.path.join(outFolder, f"{name}_all_ops.yaml")
    # with open(allOpsYAMLPath, "w") as allOpsYAML:
    # yaml.safe_dump(ops, allOpsYAML, default_flow_style=False)
    # logger.info(f"OPs yaml generated at: {allOpsYAMLPath}")

    # logger.info(f"Generating Device OPs yaml")
    # deviceOpsYAMLPath = os.path.join(outFolder, f"{name}_devices_ops.yaml")
    # with open(deviceOpsYAMLPath, "w") as deviceOpsYAML:
    # yaml.safe_dump(deviceOps, deviceOpsYAML, default_flow_style=False)
    # logger.info(f"Device OPs yaml generated at: {deviceOpsYAMLPath}")

    logger.info(f"Generating OPs CSV")
    allOpsCSVPath = os.path.join(outFolder, f"{name}.csv")
    with open(allOpsCSVPath, "w") as allOpsCSV:
        rowDicts = []

        devicePreOpTime = {}

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
                ret = ops[row]["tracy_time"]
            ret = int(ret)
            return ret

        rowKeys = list(ops.keys()) + list(signposts.keys())
        rowKeys.sort(key=row_compare)
        childCallKeys = set()
        for row in rowKeys:
            if type(row) is int:
                op = ops[row]
                if "child_calls" in op.keys():
                    for childCall in op["child_calls"]:
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
                opData = ops[op]
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

                if "kernel_info" in opData.keys():
                    rowDict["COMPUTE KERNEL PATH"] = []
                    rowDict["COMPUTE KERNEL HASH"] = []
                    rowDict["DATA MOVEMENT KERNEL PATH"] = []
                    rowDict["DATA MOVEMENT KERNEL HASH"] = []
                    for computeKernel in opData["kernel_info"]["compute_kernels"]:
                        rowDict["MATH FIDELITY"] = computeKernel["math_fidelity"]
                        rowDict["COMPUTE KERNEL PATH"].append(computeKernel["path"])
                        rowDict["COMPUTE KERNEL HASH"].append(computeKernel["name"])

                    for dmKernel in opData["kernel_info"]["datamovement_kernels"]:
                        rowDict["DATA MOVEMENT KERNEL PATH"].append(dmKernel["path"])
                        rowDict["DATA MOVEMENT KERNEL HASH"].append(dmKernel["name"])

                if "core_usage" in opData.keys():
                    rowDict["CORE COUNT"] = opData["core_usage"]["count"]

                if "device_time" in opData.keys():
                    assert "device_id" in opData.keys(), "Op has device data without device_id"
                    deviceID = opData["device_id"]
                    for analysis, analysisData in opData["device_time"].items():
                        headerField = f"{csv_header_format(analysis)} [ns]"
                        assert len(analysisData) == 1, "Unexpected device data format"
                        rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                        if analysis == "device_fw_duration":
                            rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                            rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]
                            freq = analysisData[0]["duration_cycles"] / analysisData[0]["duration_ns"]
                            if deviceID in devicePreOpTime.keys():
                                rowDict["OP TO OP LATENCY [ns]"] = round(
                                    (analysisData[0]["start_cycle"] - devicePreOpTime[deviceID]) / freq
                                )
                            else:
                                rowDict["OP TO OP LATENCY [ns]"] = 0
                            devicePreOpTime[deviceID] = analysisData[0]["end_cycle"]

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
                    rowDict["PM IDEAL [ns]"] = opData["performance_model"]["compute_ns"]
                    rowDict["PM COMPUTE [ns]"] = opData["performance_model"]["ideal_ns"]
                    rowDict["PM BANDWIDTH [ns]"] = opData["performance_model"]["bandwidth_ns"]
                    rowDict["PM REQ I BW"] = opData["performance_model"]["input_bws"]
                    rowDict["PM REQ O BW"] = opData["performance_model"]["output_bws"]

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


def process_ops(output_folder, name_append, date):
    ops, signposts = import_tracy_op_logs()

    if ops:
        deviceOps = append_device_data(ops, PROFILER_LOGS_DIR)
        generate_reports(ops, deviceOps, signposts, output_folder, date, name_append)

    else:
        deviceOps = get_device_data_generate_report(PROFILER_LOGS_DIR, output_folder, date, name_append)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
def main(output_folder, name_append, date):
    process_ops(output_folder, name_append, date)


if __name__ == "__main__":
    main()
