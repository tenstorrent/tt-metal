#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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

from tt_metal.tools.profiler.process_device_log import import_log_run_stats, generate_plots
from tt_metal.tools.profiler.process_host_log import import_host_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import (
    PROFILER_OPS_LOGS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_HOST_SIDE_LOG,
    PROFILER_OUTPUT_DIR,
)

OUT_NAME = "ops_perf_results"

OPS_CSV_HEADER = [
    "OP CODE",
    "OP TYPE",
    "GLOBAL CALL COUNT",
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
    # "CALL COUNT",
    "INPUTS",
    "OUTPUTS",
    "CALL DEPTH",
    "COMPUTE KERNEL PATH",
    "COMPUTE KERNEL HASH",
    "DATA MOVEMENT KERNEL PATH",
    "DATA MOVEMENT KERNEL HASH",
]

SORT_KEY = OPS_CSV_HEADER[2]

HOST_SIDE_STATS = ["Count", "Average"]
HOST_FUNCSTION_HEADER_FORMAT = "{} {}"

IO_FIELDS = ["W", "Z", "Y", "X", "LAYOUT", "DATA TYPE", "MEMORY"]

ttMetalFunctionsSet = set()


def parse_io_data(ioString, ioType):
    ret = {}

    IOs = ioString.split("-")
    IODict = {}
    for count, IO in enumerate(IOs):
        if IO:
            IOList = IO.split("|")
            shapeList = IOList[0].split("_")
            while len(shapeList) < 4:
                shapeList = [1] + shapeList
            IODict = {
                IO_FIELDS[0]: shapeList[0],
                IO_FIELDS[1]: shapeList[1],
                IO_FIELDS[2]: shapeList[2],
                IO_FIELDS[3]: shapeList[3],
                IO_FIELDS[4]: IOList[1],
                IO_FIELDS[5]: IOList[2],
                IO_FIELDS[6]: IOList[3],
            }
            ret[f"{ioType}_{count}"] = IODict

    return ret


def append_detail_host_time_data(opCandidatePath, call_count, timeDataDict):
    hostLogPath = os.path.join(opCandidatePath, f"{call_count}", PROFILER_HOST_SIDE_LOG)
    if os.path.isfile(hostLogPath):
        hostData = import_host_log_run_stats(hostLogPath)
        for functionName, calls in hostData.items():
            ttMetalFunctionsSet.add(functionName)
            for stat in HOST_SIDE_STATS:
                assert stat in calls["stats"].keys()
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                timeDataDict[functionKey] = int(calls["stats"][stat])


def load_device_data(opCandidatePath):
    deviceLogPath = os.path.join(opCandidatePath, PROFILER_DEVICE_SIDE_LOG)
    if os.path.isfile(deviceLogPath):
        setup = device_post_proc_config.default_setup()
        setup.intrestingCores = None
        setup.deviceInputLog = deviceLogPath
        setup.timerAnalysis = {
            "FW_START->FW_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "ANY", "timerID": 1},
                "end": {"core": "ANY", "risc": "ANY", "timerID": 4},
            },
            "KERNEL_START->KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "ANY", "timerID": 2},
                "end": {"core": "ANY", "risc": "ANY", "timerID": 3},
            },
            "BR_KERNEL_START->BR_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "BRISC", "timerID": 2},
                "end": {"core": "ANY", "risc": "BRISC", "timerID": 3},
            },
            "NC_KERNEL_START->NC_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "NCRISC", "timerID": 2},
                "end": {"core": "ANY", "risc": "NCRISC", "timerID": 3},
            },
            "T0_KERNEL_START->T0_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "TRISC_0", "timerID": 2},
                "end": {"core": "ANY", "risc": "TRISC_0", "timerID": 3},
            },
            "T1_KERNEL_START->T1_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "TRISC_1", "timerID": 2},
                "end": {"core": "ANY", "risc": "TRISC_1", "timerID": 3},
            },
            "T2_KERNEL_START->T2_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "TRISC_2", "timerID": 2},
                "end": {"core": "ANY", "risc": "TRISC_2", "timerID": 3},
            },
            "ER_KERNEL_START->ER_KERNEL_END": {
                "across": "ops",
                "type": "op_first_last",
                "start": {"core": "ANY", "risc": "ERISC", "timerID": 2},
                "end": {"core": "ANY", "risc": "ERISC", "timerID": 3},
            },
        }

        return import_log_run_stats(setup)


def append_device_time_data(call_count, timeDataDict, devicesData):
    res = True
    if devicesData and devicesData["devices"].keys():
        deviceID = list(devicesData["devices"].keys())[0]  # Assume there is only one device

        freq = devicesData["deviceInfo"]["freq"]
        opsList = devicesData["devices"][deviceID]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]

        if call_count - 1 < len(opsList):
            op = opsList[call_count - 1]
            timeseriesData = op["timeseries"]
            start_ID, start_ts, start_risc, start_core = timeseriesData[0]
            end_ID, end_ts, end_risc, end_core = timeseriesData[-1]

            cores = set()
            for timeID, ts, risc, core in timeseriesData:
                if core not in cores:
                    cores.add(core)

            deviceLevelStats = op["analysis"]

            fw_delta_time_ns = deviceLevelStats["FW_START->FW_END"]["stats"]["Average"] * 1000 / freq
            kernel_delta_time_ns = deviceLevelStats["KERNEL_START->KERNEL_END"]["stats"]["Average"] * 1000 / freq

            br_kernel_delta_time_ns = 0
            nc_kernel_delta_time_ns = 0
            t0_kernel_delta_time_ns = 0
            t1_kernel_delta_time_ns = 0
            t2_kernel_delta_time_ns = 0
            er_kernel_delta_time_ns = 0

            if "BR_KERNEL_START->BR_KERNEL_END" in deviceLevelStats.keys():
                br_kernel_delta_time_ns = (
                    deviceLevelStats["BR_KERNEL_START->BR_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
            if "NC_KERNEL_START->NC_KERNEL_END" in deviceLevelStats.keys():
                nc_kernel_delta_time_ns = (
                    deviceLevelStats["NC_KERNEL_START->NC_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
            if "T0_KERNEL_START->T0_KERNEL_END" in deviceLevelStats.keys():
                t0_kernel_delta_time_ns = (
                    deviceLevelStats["T0_KERNEL_START->T0_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
            if "T1_KERNEL_START->T1_KERNEL_END" in deviceLevelStats.keys():
                t1_kernel_delta_time_ns = (
                    deviceLevelStats["T1_KERNEL_START->T1_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
            if "T2_KERNEL_START->T2_KERNEL_END" in deviceLevelStats.keys():
                t2_kernel_delta_time_ns = (
                    deviceLevelStats["T2_KERNEL_START->T2_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
            if "ER_KERNEL_START->ER_KERNEL_END" in deviceLevelStats.keys():
                er_kernel_delta_time_ns = (
                    deviceLevelStats["ER_KERNEL_START->ER_KERNEL_END"]["stats"]["Average"] * 1000 / freq
                )
        else:
            res = False
    else:
        res = False

    if res:
        timeDataDict["DEVICE FW START CYCLE"] = start_ts
        timeDataDict["DEVICE FW END CYCLE"] = end_ts
        timeDataDict["DEVICE FW DURATION [ns]"] = round(fw_delta_time_ns)
        timeDataDict["DEVICE KERNEL DURATION [ns]"] = round(kernel_delta_time_ns)
        timeDataDict["DEVICE BRISC KERNEL DURATION [ns]"] = round(br_kernel_delta_time_ns)
        timeDataDict["DEVICE NCRISC KERNEL DURATION [ns]"] = round(nc_kernel_delta_time_ns)
        timeDataDict["DEVICE TRISC0 KERNEL DURATION [ns]"] = round(t0_kernel_delta_time_ns)
        timeDataDict["DEVICE TRISC1 KERNEL DURATION [ns]"] = round(t1_kernel_delta_time_ns)
        timeDataDict["DEVICE TRISC2 KERNEL DURATION [ns]"] = round(t2_kernel_delta_time_ns)
        timeDataDict["DEVICE ERISC KERNEL DURATION [ns]"] = round(er_kernel_delta_time_ns)
        timeDataDict["CORE COUNT"] = len(cores)
    else:
        timeDataDict["DEVICE FW START CYCLE"] = "-"
        timeDataDict["DEVICE FW END CYCLE"] = "-"
        timeDataDict["DEVICE FW DURATION [ns]"] = "-"
        timeDataDict["DEVICE KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE BRISC KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE NCRISC KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE TRISC0 KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE TRISC1 KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE TRISC2 KERNEL DURATION [ns]"] = "-"
        timeDataDict["DEVICE ERISC KERNEL DURATION [ns]"] = "-"
        timeDataDict["CORE COUNT"] = "-"


minTime = 0
maxDiff = 0
maxStackSize = 0
maxInputCount = 0
maxOutputCount = 0

op_to_folder = {}
op_flavour_to_count = {}


def parse_ops_logs(opsFolder):
    global minTime, maxDiff, maxStackSize, maxInputCount, maxOutputCount
    ops = {}

    assert os.path.isdir(opsFolder), f"{opsFolder} does no exists. Use -i option to choose the correct logs dir"
    paths = sorted(Path(opsFolder).iterdir(), key=os.path.getmtime, reverse=True)
    assert paths, f"{opsFolder} is empty. Use -i option to choose the correct logs dir"

    devicesData = load_device_data(opsFolder)

    for opCandidate in paths:
        opCandidatePath = os.path.join(opsFolder, opCandidate)
        if os.path.isdir(opCandidatePath):
            if "unknown" in str(opCandidate).lower():
                continue
            opLogPath = os.path.join(opCandidatePath, PROFILER_HOST_SIDE_LOG)
            if not os.path.isfile(opLogPath):
                logger.warning(f"Skipped: {opLogPath} dir, no host side log found.")
                continue
            with open(opLogPath, "r") as csvFile:
                csvReader = csv.DictReader(csvFile)
                for row in csvReader:
                    try:
                        op_folder_name = row["Name"].strip()
                        op_name = op_folder_name
                        extractName = re.findall(r".*tt.*tt_metal:*\d*(.*)E*", op_name)
                        if extractName:
                            op_name = extractName.pop()

                        start_ts = int(row[" Start timer count [ns]"].strip())
                        end_ts = int(row[" Stop timer count [ns]"].strip())
                        delta_time = int(row[" Delta timer count [ns]"].strip())

                        global_call_count = int(row[" Global Call Count"].strip())
                        call_count = int(row[" Call Count"].strip())
                        stack_size = int(row[" Stack Size"].strip())

                        inputs = parse_io_data(row[" Inputs"].strip(), "INPUT")
                        if len(inputs.keys()) > maxInputCount:
                            maxInputCount = len(inputs.keys())

                        outputs = parse_io_data(row[" Outputs"].strip(), "OUTPUT")
                        if len(outputs.keys()) > maxOutputCount:
                            maxOutputCount = len(outputs.keys())

                        mathFidelity = row[" Math Fidelity"].strip()
                        computeKernelPaths = row[" Compute Kernel Paths"].strip()
                        computeKernelHashes = row[" Compute Kernel Hashes"].strip()
                        datamovementKernelPaths = row[" Data Movement Kernel Paths"].strip()
                        datamovementKernelHashes = row[" Data Movement Kernel Hashes"].strip()
                        parallelizationStrategy = row[" Parallelization Strategy"].strip()
                        preferredName = row[" Preferred Name"].strip().split("tt::tt_metal::")[-1]
                        metadata = row[" Meta Data"].strip()
                        op_type = row[" Type"].strip()

                        if preferredName:
                            if op_type != "tt_dnn_device":
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
                            "OP TYPE": op_type,
                            "GLOBAL CALL COUNT": global_call_count,
                            "HOST START TS": start_ts,
                            "HOST END TS": end_ts,
                            "CALL DEPTH": stack_size,
                            "INPUTS": inputs,
                            "OUTPUTS": outputs,
                            "MATH FIDELITY": mathFidelity,
                            "COMPUTE KERNEL PATH": computeKernelPaths,
                            "COMPUTE KERNEL HASH": computeKernelHashes,
                            "DATA MOVEMENT KERNEL PATH": datamovementKernelPaths,
                            "DATA MOVEMENT KERNEL HASH": datamovementKernelHashes,
                            "PARALLELIZATION STRATEGY": parallelizationStrategy,
                            "HOST DURATION [ns]": delta_time,
                            "ATTRIBUTES": metadata,
                        }

                        append_device_time_data(global_call_count, timeDataDict, devicesData)
                        append_detail_host_time_data(opCandidatePath, call_count, timeDataDict)

                        if op_name in ops.keys():
                            ops[op_name].append(timeDataDict)
                        else:
                            ops[op_name] = [timeDataDict]
                    except KeyError as e:
                        print(e)
                        assert False, f"CSV {opLogPath} has bad header format"
    return ops


def print_ops_csv(ops, opsFolder, outputFolder, date, nameAppend):
    logger.info(f"OPs' perf analysis is finished! Generating csv ...")
    outFolder = PROFILER_OUTPUT_DIR
    if outputFolder:
        outFolder = outputFolder

    name = OUT_NAME
    outFolder = os.path.abspath(outFolder)

    if nameAppend:
        name += f"_{nameAppend}"
        outFolder = os.path.join(outFolder, nameAppend)

    testName = opsFolder.split("/")[-1]
    outFolder = os.path.join(outFolder, testName)

    if date:
        dateStr = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        name += f"_{dateStr}"
        outFolder = os.path.join(outFolder, dateStr)

    os.system(f"rm -rf {outFolder}; mkdir -p {outFolder}")

    opsCSVPath = os.path.join(outFolder, f"{name}.csv")

    if os.path.isdir(f"{opsFolder}_device"):
        os.system(
            f"mv {opsFolder} ops && mv {opsFolder}_device ops_device && tar -czf {name}.tgz ops ops_device && rm -rf ops ops_device && mv {name}.tgz {outFolder}"
        )
    else:
        os.system(f"mv {opsFolder} ops && tar -czf {name}.tgz ops && rm -rf ops && mv {name}.tgz {outFolder}")

    with open(opsCSVPath, "w") as opsCSV:
        opsWriter = csv.writer(opsCSV, delimiter=",")
        hostFunctionsHeaders = []
        for functionName in sorted(ttMetalFunctionsSet):
            for stat in HOST_SIDE_STATS:
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                if "Count" not in functionKey:
                    hostFunctionsHeaders.append(f"{functionKey} [ns]")
                else:
                    hostFunctionsHeaders.append(functionKey)

        dynamicHeader = []
        for headerItem in OPS_CSV_HEADER:
            if headerItem == "INPUTS":
                for count in range(maxInputCount):
                    for ioField in IO_FIELDS:
                        dynamicHeader.append(f"INPUT_{count}_{ioField}")
            elif headerItem == "OUTPUTS":
                for count in range(maxOutputCount):
                    for ioField in IO_FIELDS:
                        dynamicHeader.append(f"OUTPUT_{count}_{ioField}")
            else:
                dynamicHeader.append(headerItem)

        opsWriter.writerow(dynamicHeader + hostFunctionsHeaders)

        opsList = []
        for op, opCalls in ops.items():
            for opCall in opCalls:
                opCall["OP CODE"] = " " * opCall["CALL DEPTH"] + op
                opsList.append(opCall)

        opsList.sort(key=lambda item: item[SORT_KEY])
        for op in opsList:
            opRow = []
            for headerItem in dynamicHeader:
                if "INPUT" in headerItem:
                    element = "-"
                    if op["INPUTS"]:
                        io, ioField = headerItem.rsplit("_", 1)
                        if io in op["INPUTS"].keys():
                            assert ioField in op["INPUTS"][io].keys(), (headerItem, io, ioField)
                            element = op["INPUTS"][io][ioField]
                    opRow.append(element)
                elif "OUTPUT" in headerItem:
                    element = "-"
                    if op["OUTPUTS"]:
                        io, ioField = headerItem.rsplit("_", 1)
                        if io in op["OUTPUTS"].keys():
                            assert ioField in op["OUTPUTS"][io].keys(), (headerItem, io, ioField)
                            element = op["OUTPUTS"][io][ioField]
                    opRow.append(element)
                else:
                    assert headerItem in op.keys(), headerItem
                    opRow.append(op[headerItem])
            for functionName in sorted(ttMetalFunctionsSet):
                for stat in HOST_SIDE_STATS:
                    functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                    if functionKey in op.keys():
                        opRow.append(op[functionKey])
                    else:
                        opRow.append("0")
            opsWriter.writerow(opRow)

    logger.info(f"Perf CSV: {opsCSVPath}")


@click.command()
@click.option("-i", "--ops-folder", type=click.Path(exists=True, dir_okay=True), help="Ops profiler logs folder")
@click.option("-o", "--output-folder", type=click.Path(), help="Output folder for artifacts")
@click.option("-n", "--name-append", type=str, help="Name to be appended to default csv name")
@click.option("--date", default=False, is_flag=True, help="Append date to output files")
def main(ops_folder, output_folder, name_append, date):
    opsFolder = PROFILER_OPS_LOGS_DIR
    if ops_folder:
        opsFolder = os.path.abspath(ops_folder)

    ops = parse_ops_logs(opsFolder)
    print_ops_csv(ops, opsFolder, output_folder, date, name_append)


if __name__ == "__main__":
    main()
