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

import plotly.graph_objects as go
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


def import_tracy_op_logs():
    logger.info(f"Importting ops logs")
    tracyOpTimesLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_TIMES_FILE_NAME)
    tracyOpDataLog = os.path.join(PROFILER_LOGS_DIR, TRACY_OPS_DATA_FILE_NAME)

    ops = {}
    with open(tracyOpDataLog, "r") as csvFile:
        csvFile.readline()
        opsDataStrs = csvFile.read().split(";")
        opsData = []
        for opDataStr in opsDataStrs:
            if "TT_DNN" in opDataStr:
                tmpStrs = opDataStr.split("{", 1)
                if len(tmpStrs) > 1:
                    jsonStr = tmpStrs[-1]
                    jsonStr = "{" + jsonStr
                    opsData.append(json.loads(jsonStr))
    for opData in opsData:
        ops[opData["global_call_count"]] = opData

    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for op in csvReader:
            opID = int(op["zone_text"].split(":")[-1])
            assert opID in ops.keys(), f"Op time for op {opID} must present"
            ops[opID]["host_time"] = op

    return ops


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
            assert len(deviceOps[device]) == len(
                deviceOpsTime
            ), f"Device data mismatch. Expected {len(deviceOps[device])} but recieved {len(deviceOpsTime)} ops on device {device}"
            for deviceOp, deviceOpTime in zip(deviceOps[device], deviceOpsTime):
                cores = set()
                for timeID, ts, statData, risc, core in deviceOpTime["timeseries"]:
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


def generate_reports(ops, deviceOps, outputFolder, date, nameAppend):
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
        maxInputCount = -1
        maxOutputCount = -1
        inputHeaders = []
        outputHeaders = []

        def csv_header_format(header):
            return header.replace("_", " ").upper()

        for op, opData in ops.items():
            rowDict = {}
            for field, fieldData in opData.items():
                headerField = csv_header_format(field)
                if headerField in OPS_CSV_HEADER:
                    rowDict[headerField] = fieldData

            assert "host_time" in opData.keys(), "Corrupted op data"
            rowDict["HOST START TS"] = opData["host_time"]["ns_since_start"]
            rowDict["HOST END TS"] = opData["host_time"]["ns_since_start"] + opData["host_time"]["exec_time_ns"]
            rowDict["HOST DURATION [ns]"] = opData["host_time"]["exec_time_ns"]

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
                for analysis, analysisData in opData["device_time"].items():
                    headerField = f"{csv_header_format(analysis)} [ns]"
                    assert len(analysisData) == 1, "Unexpected device data format"
                    rowDict[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                rowDict["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                rowDict["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]

            assert "input_tensors" in opData.keys(), "Ops must have input tensors"
            for count, tensor in enumerate(opData["input_tensors"]):
                for ioField, ioData in tensor.items():
                    header = f"INPUT_{count}_{ioField}".upper()
                    rowDict[header] = ioData
                    if count > maxInputCount:
                        inputHeaders.append(header)
                if count > maxInputCount:
                    maxInputCount = count

            if "output_tensors" in opData.keys():
                for count, tensor in enumerate(opData["output_tensors"]):
                    for ioField, ioData in tensor.items():
                        header = f"OUTPUT_{count}_{ioField}".upper()
                        rowDict[header] = ioData
                        if count > maxOutputCount:
                            outputHeaders.append(header)
                    if count > maxOutputCount:
                        maxOutputCount = count

            if "performance_model" in opData.keys():
                rowDict["PM IDEAL [ns]"] = opData["performance_model"]["compute_ns"]
                rowDict["PM COMPUTE [ns]"] = opData["performance_model"]["ideal_ns"]
                rowDict["PM BANDWIDTH [ns]"] = opData["performance_model"]["bandwidth_ns"]
                rowDict["PM REQ I BW"] = opData["performance_model"]["input_bws"]
                rowDict["PM REQ O BW"] = opData["performance_model"]["output_bws"]

            rowDicts.append(rowDict)

        ioHeaderIndex = OPS_CSV_HEADER.index("INPUTS")
        allHeaders = OPS_CSV_HEADER[:ioHeaderIndex] + inputHeaders + outputHeaders + OPS_CSV_HEADER[ioHeaderIndex + 2 :]
        writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
        writer.writeheader()
        for rowDict in rowDicts:
            for field, fieldData in rowDict.items():
                rowDict[field] = str(fieldData).replace(",", ";")
            writer.writerow(rowDict)
    logger.info(f"OPs csv generated at: {allOpsCSVPath}")


def process_ops(output_folder, name_append, date):
    ops = import_tracy_op_logs()

    deviceOps = append_device_data(ops, PROFILER_LOGS_DIR)

    generate_reports(ops, deviceOps, output_folder, date, name_append)


@click.command()
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
def main(output_folder, name_append, date):
    process_ops(output_folder, name_append, date)


if __name__ == "__main__":
    main()
