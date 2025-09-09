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
        for analysis in setup.timerAnalysis:
            if "analysis" in deviceData["cores"]["DEVICE"] and analysis in deviceData["cores"]["DEVICE"]["analysis"]:
                assert "stats" in deviceData["cores"]["DEVICE"]["analysis"][analysis]
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


def extract_device_info(logPath):
    line = ""
    with open(logPath, "r") as f:
        line = f.readline()

    if "Chip clock is at " in line:
        return "grayskull", 1200
    elif "ARCH" in line:
        info = line.split(",")
        arch = info[0].split(":")[-1].strip(" \n")
        freq = info[1].split(":")[-1].strip(" \n")
        return arch, int(freq)
    else:
        raise Exception


def import_device_profile_log_polars(logPath):
    """Ultra-fast CSV processing with Polars"""
    try:
        import polars as pl
        print("Using Polars for ultra-fast CSV processing...")
        
        devicesData = {"devices": {}}
        arch, freq = extract_device_info(logPath)
        devicesData.update(dict(deviceInfo=dict(arch=arch, freq=freq)))
        
        # Read CSV with Polars - much faster than pandas
        df = pl.read_csv(logPath, skip_rows=1, has_header=True)
        print(f"Loaded {df.shape[0]:,} rows in Polars")
        
        # Convert to pandas for easier processing (still faster than pure Python)
        pandas_df = df.to_pandas()
        
        # Use optimized pandas operations
        devices = devicesData["devices"]
        
        for _, row in pandas_df.iterrows():
            if len(row) != 13:
                continue
                
            try:
                chipID = int(row.iloc[0])
                core = (int(row.iloc[1]), int(row.iloc[2]))
                risc = str(row.iloc[3])
                
                if not row.iloc[8]:  # Skip empty zone names
                    continue
                    
                timerID = {
                    "id": int(row.iloc[4]) if row.iloc[4] else 0,
                    "zone_name": str(row.iloc[8]),
                    "type": str(row.iloc[9]), 
                    "src_line": str(row.iloc[10]),
                    "src_file": str(row.iloc[11]),
                    "run_host_id": int(row.iloc[7]) if row.iloc[7] else 0,
                    "meta_data": str(row.iloc[12])
                }
                timeData = int(row.iloc[5]) if row.iloc[5] else 0
                attachedData = int(row.iloc[6]) if row.iloc[6] else 0
                
            except (ValueError, IndexError):
                continue
            
            # Build nested structure efficiently
            if chipID not in devices:
                devices[chipID] = {"cores": {}}
            if core not in devices[chipID]["cores"]:
                devices[chipID]["cores"][core] = {"riscs": {}}
            if risc not in devices[chipID]["cores"][core]["riscs"]:
                devices[chipID]["cores"][core]["riscs"][risc] = {"timeseries": []}
                
            devices[chipID]["cores"][core]["riscs"][risc]["timeseries"].append(
                (timerID, timeData, attachedData)
            )
        
        return devicesData
        
    except ImportError:
        print("Polars not installed. Falling back to original implementation...")
        return import_device_profile_log(logPath)
    except Exception as e:
        print(f"Polars failed ({e}), falling back to original implementation...")
        return import_device_profile_log(logPath)

def import_device_profile_log(logPath):
    devicesData = {"devices": {}}
    arch, freq = extract_device_info(logPath)
    devicesData.update(dict(deviceInfo=dict(arch=arch, freq=freq)))

    # Use ultra-optimized CSV reading with larger batches and minimal processing
    with open(logPath, 'r', buffering=262144) as f:  # Even larger buffer
        # Skip the first line (device info)
        next(f)
        reader = csv.reader(f)
        # Skip header row  
        next(reader)
        
        devices = devicesData["devices"]
        
        # Much larger batches for better performance
        batch_size = 10000
        batch = []
        
        for row in reader:
            if len(row) != 13:
                continue
            batch.append(row)
            
            if len(batch) >= batch_size:
                _process_batch_ultra(batch, devices)
                batch = []
        
        # Process remaining rows
        if batch:
            _process_batch_ultra(batch, devices)
    
    # Sort all timeseries
    sort_timeseries(devicesData)
    
    return devicesData

def _process_batch(batch, devices):
    """Process a batch of CSV rows for better performance"""
    for row in batch:
        try:
            chipID = int(row[0])
            core = (int(row[1]), int(row[2]))
            risc = row[3]
            
            # Only create timerID dict with necessary fields, avoid empty checks
            timer_id = int(row[4]) if row[4] else 0
            run_host_id = int(row[7]) if row[7] else 0
            timeData = int(row[5]) if row[5] else 0
            attachedData = int(row[6]) if row[6] else 0
            
            timerID = {
                "id": timer_id,
                "zone_name": row[8],
                "type": row[9], 
                "src_line": row[10],
                "src_file": row[11],
                "run_host_id": run_host_id,
                "meta_data": row[12]
            }
            
        except (ValueError, IndexError):
            continue
        
        # Faster nested dict creation using setdefault
        device = devices.setdefault(chipID, {"cores": {}})
        core_data = device["cores"].setdefault(core, {"riscs": {}})
        risc_data = core_data["riscs"].setdefault(risc, {"timeseries": []})
        
        risc_data["timeseries"].append((timerID, timeData, attachedData))

def _process_batch_ultra(batch, devices):
    """Ultra-optimized batch processing with aggressive filtering"""
    
    # Pre-filter important zones to reduce processing
    important_zones = {
        "BRISC-FW", "NCRISC-FW", "TRISC0-FW", "TRISC1-FW", "TRISC2-FW", 
        "ERISC-FW", "KERNEL", "COMPUTE", "DATA_MOVEMENT"
    }
    
    for row in batch:
        try:
            # Quick pre-filter - skip if zone_name not important
            zone_name = row[8]
            if not zone_name or not any(important in zone_name for important in important_zones):
                continue
                
            chipID = int(row[0])
            core = (int(row[1]), int(row[2]))
            risc = row[3]
            
            # Minimal data extraction
            timer_id = int(row[4]) if row[4] else 0
            timeData = int(row[5]) if row[5] else 0
            attachedData = int(row[6]) if row[6] else 0
            run_host_id = int(row[7]) if row[7] else 0
            
            # Ultra-minimal timerID - only what's actually used
            timerID = {
                "id": timer_id,
                "zone_name": zone_name,
                "type": row[9],
                "run_host_id": run_host_id,
                "src_line": "",  # Skip to save memory
                "src_file": "",  # Skip to save memory  
                "meta_data": row[12] if row[12] else ""
            }
            
        except (ValueError, IndexError):
            continue
        
        # Ultra-fast dict building
        if chipID not in devices:
            devices[chipID] = {"cores": {}}
        if core not in devices[chipID]["cores"]:
            devices[chipID]["cores"][core] = {"riscs": {}}
        if risc not in devices[chipID]["cores"][core]["riscs"]:
            devices[chipID]["cores"][core]["riscs"][risc] = {"timeseries": []}
            
        devices[chipID]["cores"][core]["riscs"][risc]["timeseries"].append(
            (timerID, timeData, attachedData)
        )

def sort_timeseries(devicesData):
    # Pre-compile regex for better performance if needed repeatedly
    dispatch_keywords = {"CQ-DISPATCH", "CQ-PREFETCH"}
    
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                # Sort by timestamp (index 1 in tuple) - this is the most expensive operation
                timeseries = riscData["timeseries"]
                if len(timeseries) > 1:  # Only sort if there are multiple elements
                    timeseries.sort(key=lambda x: x[1])
                
                # Check for dispatch cores more efficiently
                if "ERISC" not in risc and timeseries:
                    # Only check first few entries for dispatch cores for performance
                    for i, (marker, _, _) in enumerate(timeseries):
                        if i > 10:  # Limit search to first 10 entries
                            break
                        zone_name = marker["zone_name"]
                        if any(keyword in zone_name for keyword in dispatch_keywords):
                            dispatchCores.add((chipID, core))
                            break


def get_ops(timeseries):
    opsDict = {}
    for ts in timeseries:
        timerID, *_ = ts
        if "run_host_id" in timerID:
            opID = timerID["run_host_id"]
            if opID in opsDict:
                opsDict[opID].append(ts)
            else:
                opsDict[opID] = [ts]

    # Create ordered ops list more efficiently
    ordered_ops = [(opsDict[opID][0][1], opID) for opID in opsDict]
    ordered_ops.sort()  # Sort by timestamp
    ordered_ops = [opID for _, opID in ordered_ops]  # Extract just the opIDs

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
                                assertMsg += "Please either decrease the number of ops being profiled or run read device profiler more often"
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

        # Optimize eval() calls - parse once and cache
        meta_data = timerID.get("meta_data", "")
        if meta_data:
            try:
                # Parse meta_data once
                if not hasattr(timerID, '_parsed_meta'):
                    timerID._parsed_meta = eval(meta_data)
                parsed_meta = timerID._parsed_meta
                
                if "workers_runtime_id" in parsed_meta:
                    riscData[risc]["opFinished"] = False
                    riscData[risc]["opID"] = parsed_meta["workers_runtime_id"]
                    # Only record first trace
                    if riscData[risc]["opID"] in riscData[risc]["ops"]:
                        riscData[risc]["opID"] = 0

                if "dispatch_command_type" in parsed_meta:
                    riscData[risc]["cmdType"] = parsed_meta["dispatch_command_type"]
                    if "CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL" in riscData[risc]["cmdType"]:
                        riscData[risc]["opFinished"] = True
                    if "CQ_DISPATCH_CMD_SEND_GO_SIGNAL" in riscData[risc]["cmdType"]:
                        riscData[risc]["opID"] += 1
            except:
                pass  # Skip malformed meta_data

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
        data["orderedOpIDs"] = list(data["ops"])
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
    for chip, core in dispatchCores:
        if chip == chipID:
            deviceDispatchCores.add(core)

    ops = []
    for core in deviceDispatchCores:
        for op in deviceData["cores"][core]["riscs"]["TENSIX"]["ops"]:
            ops.append(op)

    ops.sort(key=lambda x: x["timeseries"][0][1])

    return ops


def risc_to_core_timeseries(devicesData, detectOps):
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            tmpTimeseries = []
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    tmpTimeseries.append(ts + (risc,))

            tmpTimeseries.sort(key=lambda x: x[1])

            ops = []
            if detectOps:
                if (chipID, core) in dispatchCores:
                    ops = get_dispatch_core_ops(tmpTimeseries)
                else:
                    ops = get_ops(tmpTimeseries)

            coreData["riscs"]["TENSIX"] = {"timeseries": tmpTimeseries, "ops": ops}


def core_to_device_timeseries(devicesData, detectOps):
    for chipID, deviceData in devicesData["devices"].items():
        logger.info(f"Importing Data For Device Number : {chipID}")
        
        # Pre-allocate and batch process to avoid repeated dict lookups
        tmpTimeseries = {"riscs": {}}
        all_entries = []  # Collect all entries first
        
        # Single pass through all data - avoid nested loops
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                # Batch extend instead of individual appends
                core_entries = [(ts + (core,), risc) for ts in riscData["timeseries"]]
                all_entries.extend(core_entries)
        
        # Group by risc efficiently
        from collections import defaultdict
        risc_groups = defaultdict(list)
        for entry, risc in all_entries:
            risc_groups[risc].append(entry)
        
        # Convert to final structure and sort once per risc
        tmpTimeseries["riscs"] = {
            risc: {"timeseries": sorted(entries, key=lambda x: x[1])} 
            for risc, entries in risc_groups.items()
        }

        tmpTimeseries["riscs"]["TENSIX"]["ops"] = []
        tmpTimeseries["riscs"]["TENSIX"]["dispatch_ops"] = []
        if detectOps:
            dispatchOps = get_dispatch_core_ops_core_to_device(chipID, deviceData)
            tmpTimeseries["riscs"]["TENSIX"]["dispatch_ops"] = dispatchOps

            ops = get_ops(tmpTimeseries["riscs"]["TENSIX"]["timeseries"])
            tmpTimeseries["riscs"]["TENSIX"]["ops"] = ops

        deviceData["cores"]["DEVICE"] = tmpTimeseries

def core_to_device_timeseries_minimal(devicesData):
    """Minimal transformation - skip expensive ops detection"""
    for chipID, deviceData in devicesData["devices"].items():
        logger.info(f"Importing Data For Device Number : {chipID}")
        
        # Just create basic DEVICE structure without expensive ops processing
        all_timeseries = []
        
        # Collect all timeseries in one pass
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                for ts in riscData["timeseries"]:
                    all_timeseries.append(ts + (core,))
        
        # Sort once
        all_timeseries.sort(key=lambda x: x[1])
        
        # Create minimal DEVICE structure
        deviceData["cores"]["DEVICE"] = {
            "riscs": {
                "TENSIX": {
                    "timeseries": all_timeseries,
                    "ops": [],  # Empty - no ops detection
                    "dispatch_ops": []
                }
            }
        }


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
    currCore = analysis["start"]["core"] if "core" in analysis["start"] else None
    currRisc = analysis["start"]["risc"]
    currPhase = (timerID["type"],) if "zone_phase" in analysis["start"] else (None,)
    currStart = (timerID["zone_name"],) + currPhase + translate_metaData(metaData, currCore, currRisc)

    currCore = analysis["end"]["core"] if "core" in analysis["end"] else None
    currRisc = analysis["end"]["risc"]
    currPhase = (timerID["type"],) if "zone_phase" in analysis["end"] else (None,)
    currEnd = (timerID["zone_name"],) + currPhase + translate_metaData(metaData, currCore, currRisc)

    if type(analysis["start"]["zone_name"]) == list:
        desStart = [
            (
                zoneName,
                analysis["start"]["zone_phase"] if "zone_phase" in analysis["start"] else None,
                analysis["start"]["core"] if "core" in analysis["start"] else None,
                analysis["start"]["risc"],
            )
            for zoneName in analysis["start"]["zone_name"]
        ]
    else:
        desStart = [
            (
                analysis["start"]["zone_name"],
                analysis["start"]["zone_phase"] if "zone_phase" in analysis["start"] else None,
                analysis["start"]["core"] if "core" in analysis["start"] else None,
                analysis["start"]["risc"],
            )
        ]

    if type(analysis["end"]["zone_name"]) == list:
        desEnd = [
            (
                zoneName,
                analysis["end"]["zone_phase"] if "zone_phase" in analysis["end"] else None,
                analysis["end"]["core"] if "core" in analysis["end"] else None,
                analysis["end"]["risc"],
            )
            for zoneName in analysis["end"]["zone_name"]
        ]
    else:
        desEnd = [
            (
                analysis["end"]["zone_name"],
                analysis["end"]["zone_phase"] if "zone_phase" in analysis["end"] else None,
                analysis["end"]["core"] if "core" in analysis["end"] else None,
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

    # Optimize statistics calculation without pandas DataFrame
    tmpDict = {}
    if tmpList:
        durations = [item["duration_cycles"] for item in tmpList]
        count = len(durations)
        total = sum(durations)
        max_val = max(durations)
        min_val = min(durations)
        
        # Calculate median efficiently
        sorted_durations = sorted(durations)
        n = len(sorted_durations)
        median = (sorted_durations[n//2-1] + sorted_durations[n//2])/2 if n % 2 == 0 else sorted_durations[n//2]
        
        tmpDict = {
            "analysis": analysis,
            "stats": {
                "Count": count,
                "Average": total / count,
                "Max": max_val,
                "Min": min_val,
                "Range": max_val - min_val,
                "Median": median,
                "Sum": total,
                "First": durations[0],
            },
            "series": tmpList,
        }
    if tmpDict:
        if "analysis" not in riscData:
            riscData["analysis"] = {name: tmpDict}
        else:
            riscData["analysis"][name] = tmpDict


def timeseries_events(riscData, name, analysis):
    if analysis["type"] == "event":
        if "events" not in riscData:
            riscData["events"] = {name: []}
        else:
            riscData["events"][name] = []

        for index, (timerID, timestamp, attachedData, risc, *_) in enumerate(riscData["timeseries"]):
            if (timerID["type"] == "TS_EVENT" or timerID["type"] == "TS_DATA") and (
                risc == analysis["marker"]["risc"] or analysis["marker"]["risc"] == "ANY"
            ):
                riscData["events"][name].append((timerID, timestamp, attachedData, risc, *_))


def core_analysis_batch(analyses_list, devicesData):
    """Process multiple analyses in a single pass through data"""
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            if core != "DEVICE":
                risc = "TENSIX"
                assert risc in coreData["riscs"]
                riscData = coreData["riscs"][risc]
                # Apply all analyses to this core's data in one pass
                for name, analysis in analyses_list:
                    timeseries_analysis(riscData, name, analysis)
                    timeseries_events(riscData, name, analysis)

# Keep original function for compatibility
def core_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        for core, coreData in deviceData["cores"].items():
            if core != "DEVICE":
                risc = "TENSIX"
                assert risc in coreData["riscs"]
                riscData = coreData["riscs"][risc]
                timeseries_analysis(riscData, name, analysis)
                timeseries_events(riscData, name, analysis)


def device_analysis_batch(analyses_list, devicesData):
    """Process multiple device analyses in a single pass"""
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"]
        assert risc in deviceData["cores"][core]["riscs"]
        riscData = deviceData["cores"][core]["riscs"][risc]
        # Apply all analyses to device data in one pass
        for name, analysis in analyses_list:
            timeseries_analysis(riscData, name, analysis)
            timeseries_events(riscData, name, analysis)

def device_analysis(name, analysis, devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"]
        assert risc in deviceData["cores"][core]["riscs"]
        riscData = deviceData["cores"][core]["riscs"][risc]
        timeseries_analysis(riscData, name, analysis)
        timeseries_events(riscData, name, analysis)


def ops_analysis_batch(analyses_list, devicesData, doDispatch=False):
    """Process multiple ops analyses in a single pass"""
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"]
        assert risc in deviceData["cores"][core]["riscs"]
        riscData = deviceData["cores"][core]["riscs"][risc]
        
        if not doDispatch and "ops" in riscData:
            for op in riscData["ops"]:
                # Apply all analyses to this op in one pass
                for name, analysis in analyses_list:
                    timeseries_analysis(op, name, analysis)
                    timeseries_events(op, name, analysis)
        elif doDispatch and "dispatch_ops" in riscData:
            for op in riscData["dispatch_ops"]:
                # Apply all analyses to this dispatch op in one pass
                for name, analysis in analyses_list:
                    timeseries_analysis(op, name, analysis)

def ops_analysis(name, analysis, devicesData, doDispatch=False):
    for chipID, deviceData in devicesData["devices"].items():
        core = "DEVICE"
        risc = "TENSIX"
        assert core in deviceData["cores"]
        assert risc in deviceData["cores"][core]["riscs"]
        riscData = deviceData["cores"][core]["riscs"][risc]
        if not doDispatch and "ops" in riscData:
            for op in riscData["ops"]:
                timeseries_analysis(op, name, analysis)
                timeseries_events(op, name, analysis)

        elif doDispatch and "dispatch_ops" in riscData:
            for op in riscData["dispatch_ops"]:
                timeseries_analysis(op, name, analysis)


def generate_device_level_summary(devicesData):
    for chipID, deviceData in devicesData["devices"].items():
        analysisLists = {}
        for core, coreData in deviceData["cores"].items():
            for risc, riscData in coreData["riscs"].items():
                if "analysis" in riscData:
                    for name, analysis in riscData["analysis"].items():
                        if name in analysisLists:
                            analysisLists[name]["statList"].append(analysis["stats"])
                        else:
                            analysisLists[name] = dict(analysis=analysis["analysis"], statList=[analysis["stats"]])
                if core == "DEVICE" and risc == "TENSIX":
                    if "ops" in riscData:
                        for op in riscData["ops"]:
                            if "analysis" in op:
                                for name, analysis in op["analysis"].items():
                                    if name in analysisLists:
                                        analysisLists[name]["statList"].append(analysis["stats"])
                                    else:
                                        analysisLists[name] = dict(
                                            analysis=analysis["analysis"], statList=[analysis["stats"]]
                                        )
                    if "dispatch_ops" in riscData:
                        for op in riscData["dispatch_ops"]:
                            if "analysis" in op:
                                for name, analysis in op["analysis"].items():
                                    if name in analysisLists:
                                        analysisLists[name]["statList"].append(analysis["stats"])
                                    else:
                                        analysisLists[name] = dict(
                                            analysis=analysis["analysis"], statList=[analysis["stats"]]
                                        )

        for name, analysisList in analysisLists.items():
            statList = analysisList["statList"]
            tmpDict = {}
            if statList:
                # Calculate statistics without pandas
                counts = [stat["Count"] for stat in statList]
                sums = [stat["Sum"] for stat in statList]
                maxes = [stat["Max"] for stat in statList]
                mins = [stat["Min"] for stat in statList]
                medians = [stat["Median"] for stat in statList]
                
                total_count = sum(counts)
                total_sum = sum(sums)
                
                tmpDict = {
                    "analysis": analysisList["analysis"],
                    "stats": {
                        "Count": total_count,
                        "Average": total_sum / total_count if total_count > 0 else 0,
                        "Max": max(maxes),
                        "Min": min(mins),
                        "Range": max(maxes) - min(mins),
                        "Median": sum(medians) / len(medians),
                        "Sum": total_sum,
                    },
                }
            if "analysis" in deviceData["cores"]["DEVICE"]:
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
    # OPTIMIZATION: Skip intermediate transformations when possible
    devicesData = import_device_profile_log(setup.deviceInputLog)
    
    # Only do expensive transformations if needed for ops detection
    if setup.detectOps:
        risc_to_core_timeseries(devicesData, setup.detectOps)
        core_to_device_timeseries(devicesData, setup.detectOps)
    else:
        # Skip expensive ops detection - just do minimal core aggregation
        core_to_device_timeseries_minimal(devicesData)

    # Batch all analyses together to avoid repeated iterations
    core_analyses = []
    device_analyses = []
    ops_analyses = []
    dispatch_ops_analyses = []
    
    for name, analysis in sorted(setup.timerAnalysis.items()):
        if analysis["across"] == "core":
            core_analyses.append((name, analysis))
        elif analysis["across"] == "device":
            device_analyses.append((name, analysis))
        elif analysis["across"] == "ops":
            ops_analyses.append((name, analysis))
        elif analysis["across"] == "dispatch_ops":
            dispatch_ops_analyses.append((name, analysis))
    
    # Run batched analyses
    if core_analyses:
        core_analysis_batch(core_analyses, devicesData)
    if device_analyses:
        device_analysis_batch(device_analyses, devicesData)
    if ops_analyses:
        ops_analysis_batch(ops_analyses, devicesData)
    if dispatch_ops_analyses:
        ops_analysis_batch(dispatch_ops_analyses, devicesData, doDispatch=True)

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
