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
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from math import nan, isnan
from itertools import chain

import click
from loguru import logger

from tracy.process_device_log import import_log_run_stats
from tracy.common import (
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_CPP_DEVICE_PERF_REPORT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_OUTPUT_DIR,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
    generate_logs_folder,
    generate_reports_folder,
)
from tracy import device_post_proc_config
from tracy.perf_counter_analysis import (
    extract_perf_counters,
    print_counter_statistics_summary,
    print_efficiency_metrics_summary,
    get_device_op_data,
)

yaml.SafeDumper.ignore_aliases = lambda *args: True

TRACE_OP_ID_BITSHIFT = 32

OUT_NAME = "ops_perf_results"
PER_CORE_OP_TO_OP_OUT_NAME = "per_core_op_to_op_times"
PROFILER_OP_TO_OP_OVERHEAD_NANO_SEC = 1500

OpDict = Dict[str, Any]
TraceReplayDict = Dict[int, Dict[int, List[int]]]
DeviceOpsDict = Dict[int, List[OpDict]]


OPS_CSV_HEADER = [
    "OP CODE",
    "OP TYPE",
    "GLOBAL CALL COUNT",
    "DEVICE ID",
    "DEVICE ARCH",
    "ATTRIBUTES",
    "MATH FIDELITY",
    "CORE COUNT",
    "AVAILABLE WORKER CORE COUNT",
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
    "PROGRAM HASH",
    "PROGRAM CACHE HIT",
    "TENSIX DM 0 MAX KERNEL SIZE [B]",
    "TENSIX DM 1 MAX KERNEL SIZE [B]",
    "TENSIX COMPUTE 0 MAX KERNEL SIZE [B]",
    "TENSIX COMPUTE 1 MAX KERNEL SIZE [B]",
    "TENSIX COMPUTE 2 MAX KERNEL SIZE [B]",
    "ACTIVE ETH DM 0 MAX KERNEL SIZE [B]",
    "ACTIVE ETH DM 1 MAX KERNEL SIZE [B]",
    "IDLE ETH DM 0 MAX KERNEL SIZE [B]",
    "IDLE ETH DM 1 MAX KERNEL SIZE [B]",
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM REQ I BW",
    "PM REQ O BW",
    "PM FPU UTIL (%)",
    "NOC UTIL (%)",
    "MULTICAST NOC UTIL (%)",
    "DRAM BW UTIL (%)",
    "ETH BW UTIL (%)",
    "NPE CONG IMPACT (%)",
]

# Perf counter headers are only included in CSV output when perf counter data is available.
PERF_COUNTER_CSV_HEADERS = [
    "SFPU Util Min (%)",
    "SFPU Util Median (%)",
    "SFPU Util Max (%)",
    "Avg SFPU util on full grid (%)",
    "FPU Util Min (%)",
    "FPU Util Median (%)",
    "FPU Util Max (%)",
    "Avg FPU util on full grid (%)",
    "MATH Util Min (%)",
    "MATH Util Median (%)",
    "MATH Util Max (%)",
    "Avg Math util on full grid (%)",
    "Unpacker0 Write Efficiency Min (%)",
    "Unpacker0 Write Efficiency Median (%)",
    "Unpacker0 Write Efficiency Max (%)",
    "Unpacker0 Write Efficiency Avg (%)",
    "Unpacker1 Write Efficiency Min (%)",
    "Unpacker1 Write Efficiency Median (%)",
    "Unpacker1 Write Efficiency Max (%)",
    "Unpacker1 Write Efficiency Avg (%)",
    "Unpacker Write Efficiency Min (%)",
    "Unpacker Write Efficiency Median (%)",
    "Unpacker Write Efficiency Max (%)",
    "Unpacker Write Efficiency Avg (%)",
    "Packer Efficiency Min (%)",
    "Packer Efficiency Median (%)",
    "Packer Efficiency Max (%)",
    "Packer Efficiency Avg (%)",
    "FPU Execution Efficiency Min (%)",
    "FPU Execution Efficiency Median (%)",
    "FPU Execution Efficiency Max (%)",
    "FPU Execution Efficiency Avg (%)",
    "Math Pipeline Utilization Min (%)",
    "Math Pipeline Utilization Median (%)",
    "Math Pipeline Utilization Max (%)",
    "Math Pipeline Utilization Avg (%)",
    "Math-to-Pack Handoff Efficiency Min (%)",
    "Math-to-Pack Handoff Efficiency Median (%)",
    "Math-to-Pack Handoff Efficiency Max (%)",
    "Math-to-Pack Handoff Efficiency Avg (%)",
    "Unpacker-to-Math Data Flow Min (%)",
    "Unpacker-to-Math Data Flow Median (%)",
    "Unpacker-to-Math Data Flow Max (%)",
    "Unpacker-to-Math Data Flow Avg (%)",
    # INSTRN_THREAD: Thread stall rates
    "Thread 0 Stall Rate Min (%)",
    "Thread 0 Stall Rate Median (%)",
    "Thread 0 Stall Rate Max (%)",
    "Thread 0 Stall Rate Avg (%)",
    "Thread 1 Stall Rate Min (%)",
    "Thread 1 Stall Rate Median (%)",
    "Thread 1 Stall Rate Max (%)",
    "Thread 1 Stall Rate Avg (%)",
    "Thread 2 Stall Rate Min (%)",
    "Thread 2 Stall Rate Median (%)",
    "Thread 2 Stall Rate Max (%)",
    "Thread 2 Stall Rate Avg (%)",
    # INSTRN_THREAD: Thread IPC (instructions per cycle, not %)
    "Thread 0 IPC Min",
    "Thread 0 IPC Median",
    "Thread 0 IPC Max",
    "Thread 0 IPC Avg",
    "Thread 1 IPC Min",
    "Thread 1 IPC Median",
    "Thread 1 IPC Max",
    "Thread 1 IPC Avg",
    "Thread 2 IPC Min",
    "Thread 2 IPC Median",
    "Thread 2 IPC Max",
    "Thread 2 IPC Avg",
    # INSTRN_THREAD: Pipeline wait metrics
    "SrcA Valid Wait Min (%)",
    "SrcA Valid Wait Median (%)",
    "SrcA Valid Wait Max (%)",
    "SrcA Valid Wait Avg (%)",
    "SrcB Valid Wait Min (%)",
    "SrcB Valid Wait Median (%)",
    "SrcB Valid Wait Max (%)",
    "SrcB Valid Wait Avg (%)",
    "SrcA Clear Wait Min (%)",
    "SrcA Clear Wait Median (%)",
    "SrcA Clear Wait Max (%)",
    "SrcA Clear Wait Avg (%)",
    "SrcB Clear Wait Min (%)",
    "SrcB Clear Wait Median (%)",
    "SrcB Clear Wait Max (%)",
    "SrcB Clear Wait Avg (%)",
    "Math Idle Wait T1 Min (%)",
    "Math Idle Wait T1 Median (%)",
    "Math Idle Wait T1 Max (%)",
    "Math Idle Wait T1 Avg (%)",
    "Pack Idle Wait T2 Min (%)",
    "Pack Idle Wait T2 Median (%)",
    "Pack Idle Wait T2 Max (%)",
    "Pack Idle Wait T2 Avg (%)",
    "Unpack Idle Wait T0 Min (%)",
    "Unpack Idle Wait T0 Median (%)",
    "Unpack Idle Wait T0 Max (%)",
    "Unpack Idle Wait T0 Avg (%)",
    # INSTRN_THREAD: Semaphore wait metrics
    "Semaphore Zero Wait T0 Min (%)",
    "Semaphore Zero Wait T0 Median (%)",
    "Semaphore Zero Wait T0 Max (%)",
    "Semaphore Zero Wait T0 Avg (%)",
    "Semaphore Zero Wait T1 Min (%)",
    "Semaphore Zero Wait T1 Median (%)",
    "Semaphore Zero Wait T1 Max (%)",
    "Semaphore Zero Wait T1 Avg (%)",
    "Semaphore Zero Wait T2 Min (%)",
    "Semaphore Zero Wait T2 Median (%)",
    "Semaphore Zero Wait T2 Max (%)",
    "Semaphore Zero Wait T2 Avg (%)",
    "Semaphore Full Wait T0 Min (%)",
    "Semaphore Full Wait T0 Median (%)",
    "Semaphore Full Wait T0 Max (%)",
    "Semaphore Full Wait T0 Avg (%)",
    "Semaphore Full Wait T1 Min (%)",
    "Semaphore Full Wait T1 Median (%)",
    "Semaphore Full Wait T1 Max (%)",
    "Semaphore Full Wait T1 Avg (%)",
    "Semaphore Full Wait T2 Min (%)",
    "Semaphore Full Wait T2 Median (%)",
    "Semaphore Full Wait T2 Max (%)",
    "Semaphore Full Wait T2 Avg (%)",
    # TDMA_UNPACK: Data hazard stalls
    "Data Hazard Stall Rate Min (%)",
    "Data Hazard Stall Rate Median (%)",
    "Data Hazard Stall Rate Max (%)",
    "Data Hazard Stall Rate Avg (%)",
    # L1 Bank 0: utilization metrics
    "L1 Unpacker Port Util Min (%)",
    "L1 Unpacker Port Util Median (%)",
    "L1 Unpacker Port Util Max (%)",
    "L1 Unpacker Port Util Avg (%)",
    "L1 TDMA Bundle Util Min (%)",
    "L1 TDMA Bundle Util Median (%)",
    "L1 TDMA Bundle Util Max (%)",
    "L1 TDMA Bundle Util Avg (%)",
    "NOC Ring 0 Outgoing Util Min (%)",
    "NOC Ring 0 Outgoing Util Median (%)",
    "NOC Ring 0 Outgoing Util Max (%)",
    "NOC Ring 0 Outgoing Util Avg (%)",
    "NOC Ring 0 Incoming Util Min (%)",
    "NOC Ring 0 Incoming Util Median (%)",
    "NOC Ring 0 Incoming Util Max (%)",
    "NOC Ring 0 Incoming Util Avg (%)",
    # L1 Bank 1: utilization metrics (NaN when L1_1 data unavailable)
    "NOC Ring 1 Outgoing Util Min (%)",
    "NOC Ring 1 Outgoing Util Median (%)",
    "NOC Ring 1 Outgoing Util Max (%)",
    "NOC Ring 1 Outgoing Util Avg (%)",
    "NOC Ring 1 Incoming Util Min (%)",
    "NOC Ring 1 Incoming Util Median (%)",
    "NOC Ring 1 Incoming Util Max (%)",
    "NOC Ring 1 Incoming Util Avg (%)",
]

_PERF_COUNTER_CSV_HEADERS_SET = set(PERF_COUNTER_CSV_HEADERS)


DEVICE_PERF_INT_FIELDS = {
    "GLOBAL CALL COUNT",
    "METAL TRACE ID",
    "METAL TRACE REPLAY SESSION ID",
    "DEVICE ID",
    "CORE COUNT",
    "AVAILABLE WORKER CORE COUNT",
    "DEVICE TRACE FIRMWARE DURATION [ns]",
    "DEVICE TRACE KERNEL DURATION [ns]",
    "DEVICE KERNEL FIRST TO LAST START [ns]",
    "DEVICE FW DURATION [ns]",
    "DEVICE FW START CYCLE",
    "DEVICE FW END CYCLE",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE KERNEL DURATION DM START [ns]",
    "DEVICE KERNEL START CYCLE",
    "DEVICE KERNEL END CYCLE",
    "DEVICE KERNEL DM START CYCLE",
    "DEVICE KERNEL DM END CYCLE",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
}


def _parse_int_field(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def load_device_perf_report(
    report_path: Path,
) -> Dict[int, Dict[Tuple[int, Optional[int], Optional[int]], Dict[str, Any]]]:
    """Parse cpp_device_perf_report.csv into a per-device/per-ProgramExecutionUID mapping.

    The C++ report can contain multiple rows for the same (DEVICE ID, GLOBAL CALL COUNT) when a traced program is
    replayed multiple times. Those are disambiguated by (METAL TRACE ID, METAL TRACE REPLAY SESSION ID).
    """

    per_device: Dict[int, Dict[Tuple[int, Optional[int], Optional[int]], Dict[str, Any]]] = {}
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Device perf report not found at {report_path}")

    with report_path.open("r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parsed_row: Dict[str, Any] = {}
            for header, value in row.items():
                value = value.strip() if value is not None else ""
                if header in DEVICE_PERF_INT_FIELDS:
                    parsed_row[header] = _parse_int_field(value)
                else:
                    parsed_row[header] = value

            device_id = parsed_row.get("DEVICE ID")
            op_id = parsed_row.get("GLOBAL CALL COUNT")
            trace_id = parsed_row.get("METAL TRACE ID")
            session_id = parsed_row.get("METAL TRACE REPLAY SESSION ID")
            if device_id is None or op_id is None:
                continue

            if device_id not in per_device:
                per_device[device_id] = {}
            per_device[device_id][(op_id, trace_id, session_id)] = parsed_row

    return per_device


def lookup_trace_replay_timestamp(
    traceReplays: Optional[TraceReplayDict], device_id: int, trace_id: Optional[int], session_id: Optional[int]
) -> Optional[int]:
    """Return the tracy timestamp for the requested (device, trace, session)."""

    if (
        traceReplays is None
        or trace_id is None
        or session_id is None
        or device_id not in traceReplays
        or trace_id not in traceReplays[device_id]
    ):
        return None

    timestamps = traceReplays[device_id][trace_id]
    index = session_id - 1
    if index < 0 or index >= len(timestamps):
        return None
    return timestamps[index]


def compute_ns_per_cycle(perf_row: Dict[str, Any]) -> Optional[float]:
    start_cycle = perf_row.get("DEVICE FW START CYCLE")
    end_cycle = perf_row.get("DEVICE FW END CYCLE")
    duration_ns = perf_row.get("DEVICE FW DURATION [ns]")
    if start_cycle is None or end_cycle is None or duration_ns in (None, "", 0):
        return None
    cycle_delta = end_cycle - start_cycle
    if cycle_delta <= 0:
        return None
    return duration_ns / cycle_delta


def csv_header_format(header: str) -> str:
    """Convert snake_case header strings into the canonical CSV header format."""

    return header.replace("_", " ").upper()


def import_tracy_op_logs(
    logFolder: Path,
) -> Tuple[Dict[int, OpDict], Dict[str, Dict[str, Any]], Optional[TraceReplayDict]]:
    """Parse host-side Tracy logs into per-op dictionaries, signposts, and trace replay metadata."""
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
                        try:
                            opData = json.loads(jsonStr)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping op with malformed JSON (likely truncated by Tracy's 64 KiB message limit): "
                                f"{tmpStrs[0]}"
                            )
                            continue
                        opData["metal_trace_id"] = None
                        if "op_hash" in opData:
                            assert "device_id" in opData
                            deviceID = int(opData["device_id"])
                            opHash = int(opData["op_hash"])
                            if deviceID in cached_ops:
                                cached_ops[deviceID][opHash] = opData.copy()
                            else:
                                cached_ops[deviceID] = {opHash: opData.copy()}
                            del cached_ops[deviceID][opHash]["global_call_count"]
                            if deviceID in traceIDs:
                                opData["metal_trace_id"] = traceIDs[deviceID]
                    else:  # cached device op
                        opDataList = opDataStr.split(":", 1)[-1].split(",")
                        assert len(opDataList) > 4, "Wrong cached op info format"
                        opHash = int(opDataList[1])
                        deviceID = int(opDataList[2])
                        programCacheHitStr = opDataList[3].strip()
                        programCacheHit = programCacheHitStr in ("1", "true", "True")
                        opID = int(opDataList[4])
                        if deviceID not in cached_ops or opHash not in cached_ops[deviceID]:
                            logger.warning(
                                f"Skipping cached op reference with no prior data "
                                f"(device_id={deviceID}, op_hash={opHash})"
                            )
                            continue
                        opData = cached_ops[deviceID][opHash].copy()
                        opData["global_call_count"] = opID
                        opData["program_cache_hit"] = programCacheHit
                        opData["metal_trace_id"] = None
                        if deviceID in traceIDs:
                            opData["metal_trace_id"] = traceIDs[deviceID]
                    opData["tracy_time"] = opDataTime
                    opsData.append(opData)
                elif "TRACE" in opDataStr and not opDataStr.startswith("TT_METAL_TRACE_ENQUEUE_PROGRAM"):
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

    try:
        df = pd.read_csv(tracyOpTimesLog, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_csv(tracyOpTimesLog)

    # Filter and update host_time for TT_DNN/TT_METAL ops
    # Ensure name is string type before using .str accessor
    # (pandas may infer as numeric if all values are null)
    df["name"] = df["name"].astype(str)
    tt_mask = df["name"].str.contains("TT_DNN|TT_METAL", regex=True, na=False)
    if tt_mask.any():
        tt_df = df[tt_mask]
        for op in tt_df.to_dict(orient="records"):
            opID = int(op["zone_text"].split(":")[-1])
            assert opID in ops, f"Op time for op {opID} must present. OpID: {opID}, Name: {op['name']}"
            ops[opID]["host_time"] = op

    # Similar to df["name"], ensure special_parent_text is string type before using .str accessor.
    df["special_parent_text"] = df["special_parent_text"].astype(str)
    parent_mask = df["special_parent_text"].str.contains("id:", na=False)
    if parent_mask.any():
        child_df = df[parent_mask].copy()
        child_df["parentOpID"] = child_df["special_parent_text"].str.rsplit(":", n=1).str[-1].astype(int)

        # Only process children of ops we know about
        child_df = child_df[child_df["parentOpID"].isin(ops)]

        if not child_df.empty:
            # Aggregate durations by (parentOpID, name)
            summary = child_df.groupby(["parentOpID", "name"])["exec_time_ns"].sum()
            for (pID, name), total_ns in summary.items():
                opData = ops[pID]
                if "child_calls" not in opData:
                    opData["child_calls"] = {}
                cc = opData["child_calls"]
                # Use name as key, add up total execution time
                cc[name] = cc.get(name, 0) + int(total_ns)

    return ops, signposts, traceReplays


def device_op_compare_time(op: Dict[str, Any]) -> int:
    if "timeseries" in op and len(op["timeseries"]) > 0 and len(op["timeseries"][0]) > 1:
        return int(op["timeseries"][0][1])
    else:
        return 0


def device_op_compare_opID_time(op: Dict[str, Any]) -> Tuple[int, int]:
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


def host_device_op_compare(op: OpDict) -> Tuple[int, int]:
    """Comparison key that keeps ops ordered by host id, then replay session."""

    if "metal_trace_replay_session_id" in op:
        return int(op["global_call_count"]), int(op["metal_trace_replay_session_id"])
    else:
        return int(op["global_call_count"]), 0


def extract_dispatch_op_id(dispatchOps: Dict[str, Any]) -> int:
    opId = 0
    for ts in dispatchOps["timeseries"]:
        if "meta_data" in ts[0] and "workers_runtime_id" in ts[0]["meta_data"]:
            metaData = eval(ts[0]["meta_data"])
            opId = metaData["workers_runtime_id"]
            break
    return opId


def _duplicate_series_with_ns(series: List[Dict[str, Any]], freq: int) -> List[Dict[str, Any]]:
    duplicated = []
    for entry in series:
        sample = entry.copy()
        duration_cycles = sample.get("duration_cycles")
        if duration_cycles is not None and freq:
            sample["duration_ns"] = duration_cycles * 1000 / freq
        duplicated.append(sample)
    return duplicated


def _convert_device_op_entry(device_op_time: Dict[str, Any], freq: int) -> OpDict:
    """Translate a device profiler op entry into the legacy dictionary format."""

    device_op: OpDict = {}
    cores_seen: Set[Any] = set()
    last_time_id: Optional[Dict[str, Any]] = None

    for time_id, *_rest, core in device_op_time["timeseries"]:
        last_time_id = time_id
        if "zone_name" in time_id and "FW" in time_id["zone_name"] and core not in cores_seen:
            cores_seen.add(core)

    device_op["core_usage"] = {"count": len(cores_seen), "cores": [str(core) for core in cores_seen]}
    device_op["device_time"] = {}
    for analysis, data in device_op_time["analysis"].items():
        device_op["device_time"][analysis] = {
            "series": _duplicate_series_with_ns(data["series"], freq),
            "stats": data["stats"],
        }

    if last_time_id and "run_host_id" in last_time_id:
        device_op["global_call_count"] = last_time_id["run_host_id"]
    else:
        run_host_id = device_op_time.get("op_id")
        if run_host_id is None:
            raise AssertionError("Unable to determine run_host_id for device operation entry")
        device_op["global_call_count"] = run_host_id

    return device_op


def _enrich_ops_from_perf_csv(
    host_ops_by_device: DeviceOpsDict,
    device_perf_by_device: Dict[int, Dict[Tuple[int, Optional[int], Optional[int]], Dict[str, Any]]],
    trace_replays: Optional[TraceReplayDict],
) -> DeviceOpsDict:
    for device_id in host_ops_by_device:
        assert (
            device_id in device_perf_by_device
        ), f"Device {device_id} present in host logs but missing from {PROFILER_CPP_DEVICE_PERF_REPORT}"

        # Build a lookup that matches the C++ ProgramExecutionUID structure:
        # (GLOBAL CALL COUNT, METAL TRACE ID) -> list of perf rows (one per replay session, or one for non-trace)
        perf_rows_by_key: Dict[Tuple[int, Optional[int]], List[Dict[str, Any]]] = {}
        for (op_id, trace_id, session_id), row in device_perf_by_device[device_id].items():
            perf_rows_by_key.setdefault((op_id, trace_id), []).append(row)

        enriched_ops = []
        for host_op in host_ops_by_device[device_id]:
            op_id = int(host_op["global_call_count"])
            host_trace_id = host_op.get("metal_trace_id")
            # Normalize host_trace_id: it may be None, "", or already an int
            if host_trace_id in ("", "None"):
                host_trace_id = None
            try:
                host_trace_id = int(host_trace_id) if host_trace_id is not None else None
            except (TypeError, ValueError):
                host_trace_id = None

            candidates = perf_rows_by_key.get((op_id, host_trace_id))
            if not candidates:
                # Fallback: if host didn't record trace id but perf CSV did, allow lookup by op_id only.
                candidates = []
                for (cand_op_id, _cand_trace_id), rows in perf_rows_by_key.items():
                    if cand_op_id == op_id:
                        candidates.extend(rows)

            assert candidates, (
                f"Device data missing: Op {op_id} not present in {PROFILER_CPP_DEVICE_PERF_REPORT} "
                f"for device {device_id} (trace_id={host_trace_id})"
            )

            # Create one enriched op per ProgramExecutionUID row in the C++ report.
            for perf_row in candidates:
                perf_row = perf_row.copy()
                enriched_op = copy.deepcopy(host_op)

                core_count = perf_row.get("CORE COUNT")
                if core_count is not None:
                    enriched_op["core_usage"] = {"count": core_count, "cores": []}

                metal_trace_id = perf_row.get("METAL TRACE ID")
                if metal_trace_id is not None:
                    enriched_op["metal_trace_id"] = metal_trace_id

                session_id = perf_row.get("METAL TRACE REPLAY SESSION ID")
                if session_id is not None:
                    enriched_op["metal_trace_replay_session_id"] = session_id
                    tracy_time = lookup_trace_replay_timestamp(trace_replays, device_id, metal_trace_id, session_id)
                    if tracy_time is not None:
                        enriched_op["tracy_time"] = tracy_time

                enriched_op["_device_perf_row"] = perf_row
                enriched_ops.append(enriched_op)

        host_ops_by_device[device_id] = enriched_ops
    return host_ops_by_device


def _enrich_ops_from_device_logs(
    host_ops_by_device: DeviceOpsDict,
    log_folder: Path,
    device_analysis_types: Tuple[str, ...] | List[str],
    trace_replays: Optional[TraceReplayDict],
) -> DeviceOpsDict:
    device_log_path = Path(log_folder) / PROFILER_DEVICE_SIDE_LOG
    if not device_log_path.is_file():
        raise AssertionError(
            f"{PROFILER_CPP_DEVICE_PERF_REPORT} not found and legacy device log "
            f"{PROFILER_DEVICE_SIDE_LOG} is also missing in {log_folder}."
        )

    trace_replay_counts = {}
    has_trace_runs = False
    if trace_replays:
        for device_id in trace_replays:
            trace_replay_counts[device_id] = {}
            for trace_id in trace_replays[device_id]:
                trace_replay_counts[device_id][trace_id] = len(trace_replays[device_id][trace_id])
                has_trace_runs = True

    setup = device_post_proc_config.default_setup()
    if device_analysis_types:
        available_analysis = setup.timerAnalysis
        picked_analysis = {}
        for analysis in device_analysis_types:
            assert analysis in available_analysis, f"{analysis} is not calculated in device analysis"
            picked_analysis[analysis] = available_analysis[analysis]
        setup.timerAnalysis = picked_analysis
    setup.deviceInputLog = str(device_log_path)

    device_data = import_log_run_stats(setup)
    freq = device_data["deviceInfo"]["freq"]

    for device in host_ops_by_device:
        assert device in device_data["devices"]
        device_ops_time = device_data["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
        device_dispatch_ops_time = device_data["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["dispatch_ops"]
        device_ops_time.sort(key=device_op_compare_time)

        if has_trace_runs and trace_replays:
            generated_host_data = []
            op_id_host_data_dict = {}
            for device_op in host_ops_by_device[device]:
                op_id = device_op["global_call_count"]
                assert (
                    op_id not in op_id_host_data_dict
                ), f"Host op ID cannot be repeated: op ID {op_id} was reported twice by the host"
                op_id_host_data_dict[op_id] = copy.deepcopy(device_op)

            trace_ops_map = {}
            unmatched_device_ops = []
            for device_op_time in device_ops_time:
                if len(device_op_time["timeseries"]) > 0:
                    time_id, ts, stat_data, risc, core = device_op_time["timeseries"][0]
                    assert "run_host_id" in time_id, "Device op ID missing: Device data must provide op ID"
                    device_op_id = time_id["run_host_id"]
                    if device_op_id not in op_id_host_data_dict:
                        unmatched_device_ops.append(device_op_id)
                        continue

                    trace_id = op_id_host_data_dict[device_op_id].get("metal_trace_id")
                    if trace_id is not None:
                        if device in trace_ops_map:
                            if trace_id in trace_ops_map[device]:
                                if device_op_id in trace_ops_map[device][trace_id]:
                                    trace_replays[device][trace_id].pop(0)
                                    trace_ops_map[device][trace_id] = set([device_op_id])
                                else:
                                    trace_ops_map[device][trace_id].add(device_op_id)
                            else:
                                trace_ops_map[device][trace_id] = set([device_op_id])
                        else:
                            trace_ops_map[device] = {trace_id: set([device_op_id])}

                        assert (
                            len(trace_replays[device][trace_id]) > 0
                        ), "Wrong trace replay count: Device has more ops than trace replay issued commands"

                        op_id_host_data_dict[device_op_id]["tracy_time"] = trace_replays[device][trace_id][0]
                        op_id_host_data_dict[device_op_id]["metal_trace_replay_session_id"] = (
                            trace_replay_counts[device][trace_id] - len(trace_replays[device][trace_id]) + 1
                        )
                    generated_host_data.append(copy.deepcopy(op_id_host_data_dict[device_op_id]))

            if unmatched_device_ops:
                logger.warning(
                    f"Skipping {len(unmatched_device_ops)} device op(s) with no matching host data "
                    f"on device {device} (dispatch-only trace replay entries): {unmatched_device_ops}"
                )
                matched_ids = set(op_id_host_data_dict.keys())
                device_ops_time[:] = [
                    op
                    for op in device_ops_time
                    if op["timeseries"] and op["timeseries"][0][0].get("run_host_id") in matched_ids
                ]

            # Update host_ops_by_device with generated data including trace replays
            host_ops_by_device[device] = generated_host_data

        device_ops_time.sort(key=device_op_compare_opID_time)
        host_ops_by_device[device].sort(key=host_device_op_compare)

        dispatch_op_analysis = {}
        for device_dispatch_op in device_dispatch_ops_time:
            dispatch_op_id = extract_dispatch_op_id(device_dispatch_op)
            dispatch_op_analysis[dispatch_op_id] = device_dispatch_op["analysis"]

        # attach op dispatch analysis to op analysis
        for device_op in device_ops_time:
            op_id = device_op["timeseries"][0][0]["run_host_id"]
            if op_id in dispatch_op_analysis:
                for dispatch_analysis in dispatch_op_analysis[op_id]:
                    device_op["analysis"][dispatch_analysis] = dispatch_op_analysis[op_id][dispatch_analysis]
                del dispatch_op_analysis[op_id]

        if dispatch_op_analysis:
            if has_trace_runs:
                logger.debug(
                    f"Ignoring {len(dispatch_op_analysis)} dispatch op(s) with no matching device op "
                    f"on device {device} (likely trace replay dispatch entries)"
                )
            else:
                assert False, "Unrecognized dispatch OPs are presented by dispatch cores"

        if len(host_ops_by_device[device]) != len(device_ops_time):
            device_op_id_debug = None
            host_op_id_debug = None
            for device_op, device_op_time in zip(host_ops_by_device[device], device_ops_time):
                if len(device_op_time["timeseries"]) > 0:
                    time_id, ts, stat_data, risc, core = device_op_time["timeseries"][0]
                    if "zone_name" in time_id and "FW" in time_id["zone_name"]:
                        if "run_host_id" in time_id:
                            if time_id["run_host_id"] != device_op["global_call_count"]:
                                device_op_id_debug = time_id["run_host_id"]
                                host_op_id_debug = device_op["global_call_count"]
                                break

            if device_op_id_debug and host_op_id_debug:
                assert False, (
                    f"Device data mismatch: Expected {len(host_ops_by_device[device])} "
                    f"but received {len(device_ops_time)} ops on device {device}. "
                    f"Device is showing op ID {device_op_id_debug} when host is showing op ID {host_op_id_debug}"
                )
            else:
                assert (
                    False
                ), f"Device data mismatch: Expected {len(host_ops_by_device[device])} but received {len(device_ops_time)} ops on device {device}"

        # Check if perf counters data is available
        risc_data = device_data["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]
        perf_counter_df = None
        if "events" in risc_data and "perf_counter_data" in risc_data["events"]:
            perf_counter_df = extract_perf_counters(risc_data["events"]["perf_counter_data"])

            # Print statistics for captured counter data
            if perf_counter_df is not None and not perf_counter_df.empty:
                print_counter_statistics_summary(perf_counter_df, device)

        agg_sfpu_util_min = {}
        agg_sfpu_util_median = {}
        agg_sfpu_util_max = {}
        avg_sfpu_count = {}

        agg_fpu_util_min = {}
        agg_fpu_util_median = {}
        agg_fpu_util_max = {}
        avg_fpu_count = {}

        agg_math_util_min = {}
        agg_math_util_median = {}
        agg_math_util_max = {}
        avg_math_count = {}

        agg_unpack0_eff_min = {}
        agg_unpack0_eff_median = {}
        agg_unpack0_eff_max = {}
        avg_unpack0_eff = {}

        agg_unpack1_eff_min = {}
        agg_unpack1_eff_median = {}
        agg_unpack1_eff_max = {}
        avg_unpack1_eff = {}

        agg_unpack_eff_min = {}
        agg_unpack_eff_median = {}
        agg_unpack_eff_max = {}
        avg_unpack_eff = {}

        agg_pack_eff_min = {}
        agg_pack_eff_median = {}
        agg_pack_eff_max = {}
        avg_pack_eff = {}

        agg_fpu_exec_eff_min = {}
        agg_fpu_exec_eff_median = {}
        agg_fpu_exec_eff_max = {}
        avg_fpu_exec_eff = {}

        agg_math_pipe_util_min = {}
        agg_math_pipe_util_median = {}
        agg_math_pipe_util_max = {}
        avg_math_pipe_util = {}

        agg_math_pack_eff_min = {}
        agg_math_pack_eff_median = {}
        agg_math_pack_eff_max = {}
        avg_math_pack_eff = {}

        agg_unpack_math_flow_min = {}
        agg_unpack_math_flow_median = {}
        agg_unpack_math_flow_max = {}
        avg_unpack_math_flow = {}

        if perf_counter_df is not None and not perf_counter_df.empty:
            total_compute_cores = device_data["deviceInfo"]["max_compute_cores"]

            # Helper to get counter values and ref counts by type
            def get_counter_series(counter_name):
                mask = perf_counter_df["counter type"] == counter_name
                return perf_counter_df[mask].set_index(["run_host_id", "trace_id_count", "core_x", "core_y"])["value"]

            def get_counter_ref_cnt(counter_name):
                mask = perf_counter_df["counter type"] == counter_name
                return perf_counter_df[mask].set_index(["run_host_id", "trace_id_count", "core_x", "core_y"])["ref cnt"]

            def has_counter(counter_name):
                return counter_name in perf_counter_df["counter type"].values

            def compute_util_metric(counter_name, scale=100):
                """Compute value / ref_cnt * scale per core, aggregate by op."""
                val = get_counter_series(counter_name)
                ref = get_counter_ref_cnt(counter_name)
                ratio = (val / ref * scale).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                return {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            def compute_avg_channel_util(counter_a, counter_b, scale=100):
                """Average two channel utilizations per core, then aggregate by op."""
                val_a = get_counter_series(counter_a)
                val_b = get_counter_series(counter_b)
                ref = get_counter_ref_cnt(counter_a)
                ratio = ((val_a + val_b) / 2 / ref * scale).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                return {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # Get all counter series needed for metrics
            sfpu_counter = get_counter_series("SFPU_COUNTER")
            sfpu_ref_cnt = get_counter_ref_cnt("SFPU_COUNTER")
            fpu_counter = get_counter_series("FPU_COUNTER")
            fpu_ref_cnt = get_counter_ref_cnt("FPU_COUNTER")
            math_counter = get_counter_series("MATH_COUNTER")
            math_ref_cnt = get_counter_ref_cnt("MATH_COUNTER")
            srca_write = get_counter_series("SRCA_WRITE")
            srcb_write = get_counter_series("SRCB_WRITE")
            unpack0_busy = get_counter_series("UNPACK0_BUSY_THREAD0")
            unpack1_busy = get_counter_series("UNPACK1_BUSY_THREAD0")
            srca_write_avail = get_counter_series("SRCA_WRITE_AVAILABLE")
            srcb_write_avail = get_counter_series("SRCB_WRITE_AVAILABLE")
            packer_dest_read = get_counter_series("PACKER_DEST_READ_AVAILABLE")
            packer_busy = get_counter_series("PACKER_BUSY")
            math_instrn_started = get_counter_series("MATH_INSTRN_STARTED")
            math_instrn_available = get_counter_series("MATH_INSTRN_AVAILABLE")
            available_math = get_counter_series("AVAILABLE_MATH")
            fpu_instrn_available_1 = get_counter_series("FPU_INSTRN_AVAILABLE_1")

            # Calculate utilization metrics (value / ref_cnt * 100)
            sfpu_util = (sfpu_counter / sfpu_ref_cnt * 100).replace([float("inf"), -float("inf")], nan)
            fpu_util = (fpu_counter / fpu_ref_cnt * 100).replace([float("inf"), -float("inf")], nan)
            math_util = (math_counter / math_ref_cnt * 100).replace([float("inf"), -float("inf")], nan)

            # SFPU Counter aggregations
            grouped_sfpu = sfpu_util.groupby(level=["run_host_id", "trace_id_count"])
            agg_sfpu_util_min = grouped_sfpu.min().to_dict()
            agg_sfpu_util_median = grouped_sfpu.median().to_dict()
            agg_sfpu_util_max = grouped_sfpu.max().to_dict()
            avg_sfpu_count = (
                sfpu_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
            ).to_dict()

            # FPU Counter aggregations
            grouped_fpu = fpu_util.groupby(level=["run_host_id", "trace_id_count"])
            agg_fpu_util_min = grouped_fpu.min().to_dict()
            agg_fpu_util_median = grouped_fpu.median().to_dict()
            agg_fpu_util_max = grouped_fpu.max().to_dict()
            avg_fpu_count = (
                fpu_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
            ).to_dict()

            # MATH Counter aggregations
            grouped_math = math_util.groupby(level=["run_host_id", "trace_id_count"])
            agg_math_util_min = grouped_math.min().to_dict()
            agg_math_util_median = grouped_math.median().to_dict()
            agg_math_util_max = grouped_math.max().to_dict()
            avg_math_count = (
                math_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
            ).to_dict()

            # Calculate per-core efficiency metrics
            unpack0_eff = (srca_write / unpack0_busy * 100).replace([float("inf"), -float("inf")], nan)
            unpack1_eff = (srcb_write / unpack1_busy * 100).replace([float("inf"), -float("inf")], nan)
            pack_eff = (packer_dest_read / packer_busy * 100).replace([float("inf"), -float("inf")], nan)
            math_pipe_util = (math_instrn_started / math_instrn_available * 100).replace(
                [float("inf"), -float("inf")], nan
            )
            math_pack_eff = (available_math / packer_busy * 100).replace([float("inf"), -float("inf")], nan)
            unpack_math_flow = (
                ((srca_write_avail + srcb_write_avail) / 2) / ((unpack0_busy + unpack1_busy) / 2) * 100
            ).replace([float("inf"), -float("inf")], nan)

            # Aggregate per operation (min, median, max, avg) - following same pattern as SFPU/FPU/MATH
            # Unpacker0 Write Efficiency
            grouped_unpack0 = unpack0_eff.groupby(level=["run_host_id", "trace_id_count"])
            agg_unpack0_eff_min = grouped_unpack0.min().to_dict()
            agg_unpack0_eff_median = grouped_unpack0.median().to_dict()
            agg_unpack0_eff_max = grouped_unpack0.max().to_dict()
            avg_unpack0_eff = grouped_unpack0.mean().to_dict()

            # Unpacker1 Write Efficiency
            grouped_unpack1 = unpack1_eff.groupby(level=["run_host_id", "trace_id_count"])
            agg_unpack1_eff_min = grouped_unpack1.min().to_dict()
            agg_unpack1_eff_median = grouped_unpack1.median().to_dict()
            agg_unpack1_eff_max = grouped_unpack1.max().to_dict()
            avg_unpack1_eff = grouped_unpack1.mean().to_dict()

            # Combined Unpacker Write Efficiency (average per core, then aggregate)
            unpack_combined = pd.concat([unpack0_eff, unpack1_eff], axis=1).mean(axis=1, skipna=True)
            grouped_unpack = unpack_combined.groupby(level=["run_host_id", "trace_id_count"])
            agg_unpack_eff_min = grouped_unpack.min().to_dict()
            agg_unpack_eff_median = grouped_unpack.median().to_dict()
            agg_unpack_eff_max = grouped_unpack.max().to_dict()
            avg_unpack_eff = grouped_unpack.mean().to_dict()

            # Packer Efficiency
            grouped_pack = pack_eff.groupby(level=["run_host_id", "trace_id_count"])
            agg_pack_eff_min = grouped_pack.min().to_dict()
            agg_pack_eff_median = grouped_pack.median().to_dict()
            agg_pack_eff_max = grouped_pack.max().to_dict()
            avg_pack_eff = grouped_pack.mean().to_dict()

            # FPU Execution Efficiency: FPU_COUNTER / FPU_INSTRN_AVAILABLE_1
            # Measures: when FPU work was ready (thread 1), what % actually executed?
            fpu_exec_eff = (fpu_counter / fpu_instrn_available_1 * 100).replace([float("inf"), -float("inf")], nan)
            grouped_fpu_exec = fpu_exec_eff.groupby(level=["run_host_id", "trace_id_count"])
            agg_fpu_exec_eff_min = grouped_fpu_exec.min().to_dict()
            agg_fpu_exec_eff_median = grouped_fpu_exec.median().to_dict()
            agg_fpu_exec_eff_max = grouped_fpu_exec.max().to_dict()
            avg_fpu_exec_eff = grouped_fpu_exec.mean().to_dict()

            # Math Pipeline Utilization
            grouped_math_pipe = math_pipe_util.groupby(level=["run_host_id", "trace_id_count"])
            agg_math_pipe_util_min = grouped_math_pipe.min().to_dict()
            agg_math_pipe_util_median = grouped_math_pipe.median().to_dict()
            agg_math_pipe_util_max = grouped_math_pipe.max().to_dict()
            avg_math_pipe_util = grouped_math_pipe.mean().to_dict()

            # Math-to-Pack Handoff Efficiency
            grouped_math_pack = math_pack_eff.groupby(level=["run_host_id", "trace_id_count"])
            agg_math_pack_eff_min = grouped_math_pack.min().to_dict()
            agg_math_pack_eff_median = grouped_math_pack.median().to_dict()
            agg_math_pack_eff_max = grouped_math_pack.max().to_dict()
            avg_math_pack_eff = grouped_math_pack.mean().to_dict()

            # Unpacker-to-Math Data Flow
            grouped_unpack_math = unpack_math_flow.groupby(level=["run_host_id", "trace_id_count"])
            agg_unpack_math_flow_min = grouped_unpack_math.min().to_dict()
            agg_unpack_math_flow_median = grouped_unpack_math.median().to_dict()
            agg_unpack_math_flow_max = grouped_unpack_math.max().to_dict()
            avg_unpack_math_flow = grouped_unpack_math.mean().to_dict()

            # === New metrics: INSTRN_THREAD group ===
            # Thread stall rates (value / ref_cnt * 100)
            thread_stall_metrics = {}
            for t in range(3):
                name = f"THREAD_STALLS_{t}"
                if has_counter(name):
                    thread_stall_metrics[t] = compute_util_metric(name)

            # Thread IPC (instructions / ref_cnt, no percentage scaling)
            thread_ipc_metrics = {}
            for t in range(3):
                name = f"THREAD_INSTRUCTIONS_{t}"
                if has_counter(name):
                    thread_ipc_metrics[t] = compute_util_metric(name, scale=1)

            # Pipeline wait metrics
            pipeline_wait_metrics = {}
            pipeline_wait_counters = {
                "SrcA Valid Wait": "WAITING_FOR_SRCA_VALID",
                "SrcB Valid Wait": "WAITING_FOR_SRCB_VALID",
                "SrcA Clear Wait": "WAITING_FOR_SRCA_CLEAR",
                "SrcB Clear Wait": "WAITING_FOR_SRCB_CLEAR",
                "Math Idle Wait T1": "WAITING_FOR_MATH_IDLE_1",
                "Pack Idle Wait T2": "WAITING_FOR_PACK_IDLE_2",
                "Unpack Idle Wait T0": "WAITING_FOR_UNPACK_IDLE_0",
            }
            for metric_name, counter_name in pipeline_wait_counters.items():
                if has_counter(counter_name):
                    pipeline_wait_metrics[metric_name] = compute_util_metric(counter_name)

            # Semaphore wait metrics
            sem_wait_metrics = {}
            for t in range(3):
                zero_name = f"WAITING_FOR_NONZERO_SEM_{t}"
                full_name = f"WAITING_FOR_NONFULL_SEM_{t}"
                if has_counter(zero_name):
                    sem_wait_metrics[f"Semaphore Zero Wait T{t}"] = compute_util_metric(zero_name)
                if has_counter(full_name):
                    sem_wait_metrics[f"Semaphore Full Wait T{t}"] = compute_util_metric(full_name)

            # === New metrics: TDMA_UNPACK data hazard ===
            data_hazard_metric = {}
            if has_counter("DATA_HAZARD_STALLS_MOVD2A"):
                data_hazard_metric = compute_util_metric("DATA_HAZARD_STALLS_MOVD2A")

            # === New metrics: L1 Bank 0 ===
            l1_unpacker_util = {}
            l1_tdma_bundle_util = {}
            noc_r0_out_util = {}
            noc_r0_in_util = {}
            if has_counter("L1_0_UNPACKER_0"):
                l1_unpacker_util = compute_util_metric("L1_0_UNPACKER_0")
            if has_counter("L1_0_TDMA_BUNDLE_0_RISC") and has_counter("L1_0_TDMA_BUNDLE_1_TRISC"):
                l1_tdma_bundle_util = compute_avg_channel_util("L1_0_TDMA_BUNDLE_0_RISC", "L1_0_TDMA_BUNDLE_1_TRISC")
            if has_counter("L1_0_NOC_RING0_OUTGOING_0") and has_counter("L1_0_NOC_RING0_OUTGOING_1"):
                noc_r0_out_util = compute_avg_channel_util("L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1")
            if has_counter("L1_0_NOC_RING0_INCOMING_0") and has_counter("L1_0_NOC_RING0_INCOMING_1"):
                noc_r0_in_util = compute_avg_channel_util("L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1")

            # === New metrics: L1 Bank 1 ===
            noc_r1_out_util = {}
            noc_r1_in_util = {}
            if has_counter("L1_1_NOC_RING1_OUTGOING_0") and has_counter("L1_1_NOC_RING1_OUTGOING_1"):
                noc_r1_out_util = compute_avg_channel_util("L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1")
            if has_counter("L1_1_NOC_RING1_INCOMING_0") and has_counter("L1_1_NOC_RING1_INCOMING_1"):
                noc_r1_in_util = compute_avg_channel_util("L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1")

        # Enrich ops with device data and perf counters
        for device_op, device_op_time in zip(host_ops_by_device[device], device_ops_time):
            # Verify match again (redundant but safe)
            if len(device_op_time["timeseries"]) > 0:
                time_id = device_op_time["timeseries"][0][0]
                if "run_host_id" in time_id:
                    assert time_id["run_host_id"] == device_op["global_call_count"]

            # Extract basic device data
            legacy_data = _convert_device_op_entry(device_op_time, freq)
            device_op.update(legacy_data)

            # Add perf counters
            trace_id_counter = device_op.get("metal_trace_replay_session_id", -1)
            global_call_count = device_op["global_call_count"]
            device_op["freq"] = freq

            if perf_counter_df is not None and not perf_counter_df.empty:
                lookup_key = (global_call_count, trace_id_counter)
                # SFPU
                sfpu_min_val = agg_sfpu_util_min.get(lookup_key, nan)
                sfpu_median_val = agg_sfpu_util_median.get(lookup_key, nan)
                sfpu_max_val = agg_sfpu_util_max.get(lookup_key, nan)
                device_op["SFPU Util Min (%)"] = sfpu_min_val
                device_op["SFPU Util Median (%)"] = sfpu_median_val
                device_op["SFPU Util Max (%)"] = sfpu_max_val

                # FPU
                fpu_min_val = agg_fpu_util_min.get(lookup_key, nan)
                fpu_median_val = agg_fpu_util_median.get(lookup_key, nan)
                fpu_max_val = agg_fpu_util_max.get(lookup_key, nan)
                device_op["FPU Util Min (%)"] = fpu_min_val
                device_op["FPU Util Median (%)"] = fpu_median_val
                device_op["FPU Util Max (%)"] = fpu_max_val

                # MATH
                math_min_val = agg_math_util_min.get(lookup_key, nan)
                math_median_val = agg_math_util_median.get(lookup_key, nan)
                math_max_val = agg_math_util_max.get(lookup_key, nan)
                device_op["MATH Util Min (%)"] = math_min_val
                device_op["MATH Util Median (%)"] = math_median_val
                device_op["MATH Util Max (%)"] = math_max_val

                device_op["avg_sfpu_count"] = avg_sfpu_count.get(lookup_key, nan)
                device_op["avg_fpu_count"] = avg_fpu_count.get(lookup_key, nan)
                device_op["avg_math_count"] = avg_math_count.get(lookup_key, nan)

                # Unpacker0 Write Efficiency
                device_op["Unpacker0 Write Efficiency Min (%)"] = agg_unpack0_eff_min.get(lookup_key, nan)
                device_op["Unpacker0 Write Efficiency Median (%)"] = agg_unpack0_eff_median.get(lookup_key, nan)
                device_op["Unpacker0 Write Efficiency Max (%)"] = agg_unpack0_eff_max.get(lookup_key, nan)
                device_op["Unpacker0 Write Efficiency Avg (%)"] = avg_unpack0_eff.get(lookup_key, nan)

                # Unpacker1 Write Efficiency
                device_op["Unpacker1 Write Efficiency Min (%)"] = agg_unpack1_eff_min.get(lookup_key, nan)
                device_op["Unpacker1 Write Efficiency Median (%)"] = agg_unpack1_eff_median.get(lookup_key, nan)
                device_op["Unpacker1 Write Efficiency Max (%)"] = agg_unpack1_eff_max.get(lookup_key, nan)
                device_op["Unpacker1 Write Efficiency Avg (%)"] = avg_unpack1_eff.get(lookup_key, nan)

                # Combined Unpacker Write Efficiency
                device_op["Unpacker Write Efficiency Min (%)"] = agg_unpack_eff_min.get(lookup_key, nan)
                device_op["Unpacker Write Efficiency Median (%)"] = agg_unpack_eff_median.get(lookup_key, nan)
                device_op["Unpacker Write Efficiency Max (%)"] = agg_unpack_eff_max.get(lookup_key, nan)
                device_op["Unpacker Write Efficiency Avg (%)"] = avg_unpack_eff.get(lookup_key, nan)

                # Packer Efficiency
                device_op["Packer Efficiency Min (%)"] = agg_pack_eff_min.get(lookup_key, nan)
                device_op["Packer Efficiency Median (%)"] = agg_pack_eff_median.get(lookup_key, nan)
                device_op["Packer Efficiency Max (%)"] = agg_pack_eff_max.get(lookup_key, nan)
                device_op["Packer Efficiency Avg (%)"] = avg_pack_eff.get(lookup_key, nan)

                # FPU Execution Efficiency
                device_op["FPU Execution Efficiency Min (%)"] = agg_fpu_exec_eff_min.get(lookup_key, nan)
                device_op["FPU Execution Efficiency Median (%)"] = agg_fpu_exec_eff_median.get(lookup_key, nan)
                device_op["FPU Execution Efficiency Max (%)"] = agg_fpu_exec_eff_max.get(lookup_key, nan)
                device_op["FPU Execution Efficiency Avg (%)"] = avg_fpu_exec_eff.get(lookup_key, nan)

                # Math Pipeline Utilization
                device_op["Math Pipeline Utilization Min (%)"] = agg_math_pipe_util_min.get(lookup_key, nan)
                device_op["Math Pipeline Utilization Median (%)"] = agg_math_pipe_util_median.get(lookup_key, nan)
                device_op["Math Pipeline Utilization Max (%)"] = agg_math_pipe_util_max.get(lookup_key, nan)
                device_op["Math Pipeline Utilization Avg (%)"] = avg_math_pipe_util.get(lookup_key, nan)

                # Math-to-Pack Handoff Efficiency
                device_op["Math-to-Pack Handoff Efficiency Min (%)"] = agg_math_pack_eff_min.get(lookup_key, nan)
                device_op["Math-to-Pack Handoff Efficiency Median (%)"] = agg_math_pack_eff_median.get(lookup_key, nan)
                device_op["Math-to-Pack Handoff Efficiency Max (%)"] = agg_math_pack_eff_max.get(lookup_key, nan)
                device_op["Math-to-Pack Handoff Efficiency Avg (%)"] = avg_math_pack_eff.get(lookup_key, nan)

                # Unpacker-to-Math Data Flow
                device_op["Unpacker-to-Math Data Flow Min (%)"] = agg_unpack_math_flow_min.get(lookup_key, nan)
                device_op["Unpacker-to-Math Data Flow Median (%)"] = agg_unpack_math_flow_median.get(lookup_key, nan)
                device_op["Unpacker-to-Math Data Flow Max (%)"] = agg_unpack_math_flow_max.get(lookup_key, nan)
                device_op["Unpacker-to-Math Data Flow Avg (%)"] = avg_unpack_math_flow.get(lookup_key, nan)

                # Helper to assign a metric dict's 4 stats to device_op
                def assign_metric(base_name, metric_dict, suffix=" (%)", lookup=lookup_key):
                    if metric_dict:
                        device_op[f"{base_name} Min{suffix}"] = metric_dict["min"].get(lookup, nan)
                        device_op[f"{base_name} Median{suffix}"] = metric_dict["median"].get(lookup, nan)
                        device_op[f"{base_name} Max{suffix}"] = metric_dict["max"].get(lookup, nan)
                        device_op[f"{base_name} Avg{suffix}"] = metric_dict["avg"].get(lookup, nan)

                # Thread stall rates
                for t in range(3):
                    assign_metric(f"Thread {t} Stall Rate", thread_stall_metrics.get(t, {}))

                # Thread IPC (not percentage)
                for t in range(3):
                    assign_metric(f"Thread {t} IPC", thread_ipc_metrics.get(t, {}), suffix="")

                # Pipeline wait metrics
                for metric_name, metric_data in pipeline_wait_metrics.items():
                    assign_metric(metric_name, metric_data)

                # Semaphore wait metrics
                for metric_name, metric_data in sem_wait_metrics.items():
                    assign_metric(metric_name, metric_data)

                # Data Hazard Stall Rate
                assign_metric("Data Hazard Stall Rate", data_hazard_metric)

                # L1 Bank 0 metrics
                assign_metric("L1 Unpacker Port Util", l1_unpacker_util)
                assign_metric("L1 TDMA Bundle Util", l1_tdma_bundle_util)
                assign_metric("NOC Ring 0 Outgoing Util", noc_r0_out_util)
                assign_metric("NOC Ring 0 Incoming Util", noc_r0_in_util)

                # L1 Bank 1 metrics
                assign_metric("NOC Ring 1 Outgoing Util", noc_r1_out_util)
                assign_metric("NOC Ring 1 Incoming Util", noc_r1_in_util)

        if perf_counter_df is not None and not perf_counter_df.empty:
            print_efficiency_metrics_summary(pd.DataFrame(host_ops_by_device[device]), device)

    return host_ops_by_device


def _build_trace_ops_mapping(host_ops_by_device: DeviceOpsDict, ops: Dict[int, OpDict]) -> Dict[int, OpDict]:
    trace_ops_by_augmented_id: Dict[int, OpDict] = {}
    for _, per_device_ops in host_ops_by_device.items():
        for op in per_device_ops:
            if "metal_trace_replay_session_id" in op:
                augmented_id = op["global_call_count"] | (op["metal_trace_replay_session_id"] << TRACE_OP_ID_BITSHIFT)
                trace_copy = copy.deepcopy(op)
                trace_copy["global_call_count"] = augmented_id
                trace_ops_by_augmented_id[augmented_id] = trace_copy
            else:
                ops[op["global_call_count"]] = op
    return trace_ops_by_augmented_id


# Append device data to device ops and return the list of mapped device op ref list
def append_device_data(
    ops: Dict[int, OpDict],
    traceReplays: Optional[TraceReplayDict],
    logFolder: Path,
    analyze_noc_traces: bool,
    device_analysis_types: Tuple[str, ...] | List[str],
    force_legacy_device_logs: bool = False,
) -> Tuple[DeviceOpsDict, Dict[int, OpDict]]:
    """Join host metadata with either the perf CSV or legacy device logs."""

    host_ops_by_device, _ = get_device_op_data(ops, host_device_op_compare)
    logger.info("Appending device data")

    device_perf_report = Path(logFolder) / PROFILER_CPP_DEVICE_PERF_REPORT
    use_perf_csv = device_perf_report.is_file() and not force_legacy_device_logs

    if use_perf_csv:
        if device_analysis_types:
            logger.warning(
                "device_analysis_types is not supported when using cpp_device_perf_report.csv; ignoring option."
            )
        device_perf_by_device = load_device_perf_report(device_perf_report)
        host_ops_by_device = _enrich_ops_from_perf_csv(host_ops_by_device, device_perf_by_device, traceReplays)
    else:
        if device_perf_report.is_file() and force_legacy_device_logs:
            logger.info(
                f"Forcing legacy device-log parsing even though {PROFILER_CPP_DEVICE_PERF_REPORT} exists in {logFolder}."
            )
        else:
            logger.warning(
                f"Device perf report {PROFILER_CPP_DEVICE_PERF_REPORT} not found in {logFolder}. "
                f"Falling back to legacy device-log parsing via import_log_run_stats(); this will take longer."
            )
        # Pass traceReplays so legacy path can generate trace host data
        host_ops_by_device = _enrich_ops_from_device_logs(
            host_ops_by_device, logFolder, device_analysis_types, traceReplays
        )

    trace_ops_by_augmented_id = _build_trace_ops_mapping(host_ops_by_device, ops)

    if analyze_noc_traces:
        npe_stats = analyzeNoCTraces(logFolder)
        if npe_stats is not None:
            ops_found = 0
            for op in chain(*host_ops_by_device.values(), trace_ops_by_augmented_id.values()):
                global_call_count = op["global_call_count"] & ((1 << TRACE_OP_ID_BITSHIFT) - 1)
                metal_trace_id = op.get("metal_trace_id", None)
                metal_trace_replay_session_id = op.get("metal_trace_replay_session_id", None)
                op_npe_stats = npe_stats.getDatapointByID(
                    global_call_count, metal_trace_id, metal_trace_replay_session_id
                )
                if op_npe_stats is not None:
                    ops_found += 1
                    op["NOC UTIL (%)"] = round(op_npe_stats.result.overall_avg_link_util, 1)
                    op["MULTICAST NOC UTIL (%)"] = round(op_npe_stats.result.overall_avg_mcast_write_link_util, 1)
                    op["DRAM BW UTIL (%)"] = round(op_npe_stats.result.dram_bw_util, 1)
                    op["ETH BW UTIL (%)"] = op_npe_stats.result.getEthBwUtilPerCoreStr()
                    op["NPE CONG IMPACT (%)"] = round(op_npe_stats.result.getCongestionImpact(), 2)
            logger.info(f"Analyzed {ops_found} operations with tt-npe trace data.")

    return host_ops_by_device, trace_ops_by_augmented_id


def get_device_data_generate_report(
    logFolder: Path,
    outputFolder: Optional[Path],
    date: bool,
    nameAppend: Optional[str],
    export_csv: bool = True,
    cleanup_device_log: bool = False,
    device_analysis_types: Tuple[str, ...] | List[str] = (),
) -> List[Dict[str, Any]]:
    """Generate CSV rows using only device-side logs (no host metadata)."""

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
                assert analysis in allAnalysis, f" {analysis} is not calculated in device analysis"
                pickedAnalysis[analysis] = allAnalysis[analysis]

            setup.timerAnalysis = pickedAnalysis
        setup.deviceInputLog = deviceTimesLog
        deviceData = import_log_run_stats(setup)
        logger.info(f"Generating device op report ...")
        freq = deviceData["deviceInfo"]["freq"]

        # Calculate efficiency metrics for all devices (device-only mode)
        device_efficiency_metrics = {}
        for device in deviceData["devices"]:
            risc_data = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]
            if "events" in risc_data and "perf_counter_data" in risc_data["events"]:
                perf_counter_df = extract_perf_counters(risc_data["events"]["perf_counter_data"])

                if perf_counter_df is not None and not perf_counter_df.empty:
                    # Print statistics for captured counter data
                    print_counter_statistics_summary(perf_counter_df, device)

                    # Calculate efficiency metrics for this device
                    import pandas as pd
                    from math import nan

                    # Create efficiency dataframe
                    efficiency_records = []
                    for _, row in perf_counter_df.iterrows():
                        efficiency_records.append(
                            {
                                "run_host_id": row["run_host_id"],
                                "trace_id_count": row["trace_id_count"],
                                "core_x": row["core_x"],
                                "core_y": row["core_y"],
                                "counter_type": row["counter type"],
                                "value": row["value"],
                                "ref_cnt": row["ref cnt"],
                            }
                        )

                    eff_df = pd.DataFrame(efficiency_records)

                    # Pivot to get all counter types per (op, core)
                    eff_pivot = eff_df.pivot_table(
                        index=["run_host_id", "trace_id_count", "core_x", "core_y"],
                        columns="counter_type",
                        values=["value", "ref_cnt"],
                        aggfunc="first",
                    ).reset_index()

                    # Flatten column names
                    eff_pivot.columns = [
                        "_".join(col).strip("_") if col[1] else col[0] for col in eff_pivot.columns.values
                    ]

                    # Helper function for safe division
                    def safe_div(num, denom):
                        return (num / denom * 100) if denom > 0 else nan

                    # Calculate per-core efficiency metrics
                    eff_pivot["SFPU Util"] = eff_pivot.apply(
                        lambda x: (x.get("value_SFPU_COUNTER", 0) / x.get("ref_cnt_SFPU_COUNTER", 1) * 100)
                        if x.get("ref_cnt_SFPU_COUNTER", 0) > 0
                        else nan,
                        axis=1,
                    )
                    eff_pivot["FPU Util"] = eff_pivot.apply(
                        lambda x: (x.get("value_FPU_COUNTER", 0) / x.get("ref_cnt_FPU_COUNTER", 1) * 100)
                        if x.get("ref_cnt_FPU_COUNTER", 0) > 0
                        else nan,
                        axis=1,
                    )
                    eff_pivot["MATH Util"] = eff_pivot.apply(
                        lambda x: (x.get("value_MATH_COUNTER", 0) / x.get("ref_cnt_MATH_COUNTER", 1) * 100)
                        if x.get("ref_cnt_MATH_COUNTER", 0) > 0
                        else nan,
                        axis=1,
                    )
                    eff_pivot["Unpacker0 Write Efficiency"] = eff_pivot.apply(
                        lambda x: safe_div(x.get("value_SRCA_WRITE", 0), x.get("value_UNPACK0_BUSY_THREAD0", 0)), axis=1
                    )
                    eff_pivot["Unpacker1 Write Efficiency"] = eff_pivot.apply(
                        lambda x: safe_div(x.get("value_SRCB_WRITE", 0), x.get("value_UNPACK1_BUSY_THREAD0", 0)), axis=1
                    )
                    eff_pivot["Packer Efficiency"] = eff_pivot.apply(
                        lambda x: safe_div(x.get("value_PACKER_DEST_READ_AVAILABLE", 0), x.get("value_PACKER_BUSY", 0)),
                        axis=1,
                    )
                    eff_pivot["Math Pipeline Utilization"] = eff_pivot.apply(
                        lambda x: safe_div(
                            x.get("value_MATH_INSTRN_STARTED", 0), x.get("value_MATH_INSTRN_AVAILABLE", 0)
                        ),
                        axis=1,
                    )
                    eff_pivot["Math-to-Pack Handoff Efficiency"] = eff_pivot.apply(
                        lambda x: safe_div(x.get("value_AVAILABLE_MATH", 0), x.get("value_PACKER_BUSY", 0)), axis=1
                    )
                    eff_pivot["Unpacker-to-Math Data Flow"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_SRCA_WRITE_AVAILABLE", 0) + x.get("value_SRCB_WRITE_AVAILABLE", 0)) / 2,
                            (x.get("value_UNPACK0_BUSY_THREAD0", 0) + x.get("value_UNPACK1_BUSY_THREAD0", 0)) / 2,
                        ),
                        axis=1,
                    )
                    eff_pivot["Unpacker Write Efficiency"] = eff_pivot[
                        ["Unpacker0 Write Efficiency", "Unpacker1 Write Efficiency"]
                    ].mean(axis=1, skipna=True)
                    eff_pivot["FPU Execution Efficiency"] = eff_pivot.apply(
                        lambda x: (x.get("value_FPU_COUNTER", 0) / x.get("value_FPU_INSTRN_AVAILABLE_1", 1) * 100)
                        if x.get("value_FPU_INSTRN_AVAILABLE_1", 0) > 0
                        else nan,
                        axis=1,
                    )

                    # New metrics: Thread stall rates and IPC
                    for t in range(3):
                        stall_col = f"value_THREAD_STALLS_{t}"
                        ipc_col = f"value_THREAD_INSTRUCTIONS_{t}"
                        ref_col = f"ref_cnt_THREAD_STALLS_{t}"
                        eff_pivot[f"Thread {t} Stall Rate"] = eff_pivot.apply(
                            lambda x, s=stall_col, r=ref_col: safe_div(x.get(s, 0), x.get(r, 0)), axis=1
                        )
                        ref_ipc = f"ref_cnt_THREAD_INSTRUCTIONS_{t}"
                        eff_pivot[f"Thread {t} IPC"] = eff_pivot.apply(
                            lambda x, v=ipc_col, r=ref_ipc: (x.get(v, 0) / x.get(r, 1)) if x.get(r, 0) > 0 else nan,
                            axis=1,
                        )

                    # Pipeline wait metrics
                    pipeline_wait_defs = {
                        "SrcA Valid Wait": "WAITING_FOR_SRCA_VALID",
                        "SrcB Valid Wait": "WAITING_FOR_SRCB_VALID",
                        "SrcA Clear Wait": "WAITING_FOR_SRCA_CLEAR",
                        "SrcB Clear Wait": "WAITING_FOR_SRCB_CLEAR",
                        "Math Idle Wait T1": "WAITING_FOR_MATH_IDLE_1",
                        "Pack Idle Wait T2": "WAITING_FOR_PACK_IDLE_2",
                        "Unpack Idle Wait T0": "WAITING_FOR_UNPACK_IDLE_0",
                    }
                    for metric_name, counter_name in pipeline_wait_defs.items():
                        val_col = f"value_{counter_name}"
                        ref_col = f"ref_cnt_{counter_name}"
                        eff_pivot[metric_name] = eff_pivot.apply(
                            lambda x, v=val_col, r=ref_col: safe_div(x.get(v, 0), x.get(r, 0)), axis=1
                        )

                    # Semaphore wait metrics
                    for t in range(3):
                        for kind, prefix in [
                            ("Semaphore Zero Wait", "WAITING_FOR_NONZERO_SEM"),
                            ("Semaphore Full Wait", "WAITING_FOR_NONFULL_SEM"),
                        ]:
                            val_col = f"value_{prefix}_{t}"
                            ref_col = f"ref_cnt_{prefix}_{t}"
                            eff_pivot[f"{kind} T{t}"] = eff_pivot.apply(
                                lambda x, v=val_col, r=ref_col: safe_div(x.get(v, 0), x.get(r, 0)), axis=1
                            )

                    # Data Hazard Stall Rate
                    eff_pivot["Data Hazard Stall Rate"] = eff_pivot.apply(
                        lambda x: safe_div(
                            x.get("value_DATA_HAZARD_STALLS_MOVD2A", 0), x.get("ref_cnt_DATA_HAZARD_STALLS_MOVD2A", 0)
                        ),
                        axis=1,
                    )

                    # L1 Bank 0 metrics
                    eff_pivot["L1 Unpacker Port Util"] = eff_pivot.apply(
                        lambda x: safe_div(x.get("value_L1_0_UNPACKER_0", 0), x.get("ref_cnt_L1_0_UNPACKER_0", 0)),
                        axis=1,
                    )
                    eff_pivot["L1 TDMA Bundle Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_L1_0_TDMA_BUNDLE_0_RISC", 0) + x.get("value_L1_0_TDMA_BUNDLE_1_TRISC", 0))
                            / 2,
                            x.get("ref_cnt_L1_0_TDMA_BUNDLE_0_RISC", 0),
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 0 Outgoing Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0))
                            / 2,
                            x.get("ref_cnt_L1_0_NOC_RING0_OUTGOING_0", 0),
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 0 Incoming Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0))
                            / 2,
                            x.get("ref_cnt_L1_0_NOC_RING0_INCOMING_0", 0),
                        ),
                        axis=1,
                    )
                    # L1 Bank 1 metrics
                    eff_pivot["NOC Ring 1 Outgoing Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_L1_1_NOC_RING1_OUTGOING_0", 0) + x.get("value_L1_1_NOC_RING1_OUTGOING_1", 0))
                            / 2,
                            x.get("ref_cnt_L1_1_NOC_RING1_OUTGOING_0", 0),
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 1 Incoming Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            (x.get("value_L1_1_NOC_RING1_INCOMING_0", 0) + x.get("value_L1_1_NOC_RING1_INCOMING_1", 0))
                            / 2,
                            x.get("ref_cnt_L1_1_NOC_RING1_INCOMING_0", 0),
                        ),
                        axis=1,
                    )

                    # Aggregate metrics per operation (min, median, max, avg)
                    grouped_eff = eff_pivot.groupby(["run_host_id", "trace_id_count"])

                    # Store all aggregated metrics for this device using a systematic approach
                    # All metric base names that use (%) suffix
                    _pct_metric_names = [
                        "SFPU Util",
                        "FPU Util",
                        "MATH Util",
                        "Unpacker0 Write Efficiency",
                        "Unpacker1 Write Efficiency",
                        "Unpacker Write Efficiency",
                        "Packer Efficiency",
                        "FPU Execution Efficiency",
                        "Math Pipeline Utilization",
                        "Math-to-Pack Handoff Efficiency",
                        "Unpacker-to-Math Data Flow",
                        "Thread 0 Stall Rate",
                        "Thread 1 Stall Rate",
                        "Thread 2 Stall Rate",
                        "SrcA Valid Wait",
                        "SrcB Valid Wait",
                        "SrcA Clear Wait",
                        "SrcB Clear Wait",
                        "Math Idle Wait T1",
                        "Pack Idle Wait T2",
                        "Unpack Idle Wait T0",
                        "Semaphore Zero Wait T0",
                        "Semaphore Zero Wait T1",
                        "Semaphore Zero Wait T2",
                        "Semaphore Full Wait T0",
                        "Semaphore Full Wait T1",
                        "Semaphore Full Wait T2",
                        "Data Hazard Stall Rate",
                        "L1 Unpacker Port Util",
                        "L1 TDMA Bundle Util",
                        "NOC Ring 0 Outgoing Util",
                        "NOC Ring 0 Incoming Util",
                        "NOC Ring 1 Outgoing Util",
                        "NOC Ring 1 Incoming Util",
                    ]
                    # IPC metrics (no % suffix)
                    _ipc_metric_names = ["Thread 0 IPC", "Thread 1 IPC", "Thread 2 IPC"]

                    agg_metrics = {}
                    for base_name in _pct_metric_names + _ipc_metric_names:
                        if base_name in eff_pivot.columns:
                            agg_metrics[base_name] = {
                                "min": grouped_eff[base_name].min().to_dict(),
                                "median": grouped_eff[base_name].median().to_dict(),
                                "max": grouped_eff[base_name].max().to_dict(),
                                "avg": grouped_eff[base_name].mean().to_dict(),
                            }
                    device_efficiency_metrics[device] = agg_metrics

                    # Print efficiency summary
                    eff_summary_df = []
                    first_metric = next(iter(agg_metrics.values()), {})
                    first_stat = first_metric.get("min", {})
                    for key in first_stat.keys():
                        row = {}
                        for base_name in _pct_metric_names:
                            if base_name in agg_metrics:
                                m = agg_metrics[base_name]
                                for stat in ["min", "median", "max", "avg"]:
                                    stat_cap = stat.capitalize() if stat != "avg" else "Avg"
                                    row[f"{base_name} {stat_cap} (%)"] = m[stat].get(key, nan)
                        for base_name in _ipc_metric_names:
                            if base_name in agg_metrics:
                                m = agg_metrics[base_name]
                                for stat in ["min", "median", "max", "avg"]:
                                    stat_cap = stat.capitalize() if stat != "avg" else "Avg"
                                    row[f"{base_name} {stat_cap}"] = m[stat].get(key, nan)
                        eff_summary_df.append(row)
                    if eff_summary_df:
                        print_efficiency_metrics_summary(pd.DataFrame(eff_summary_df), device)

        for device in deviceData["devices"]:
            deviceOps[device] = []
            deviceOpsTime = deviceData["devices"][device]["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
            for deviceOpTime in deviceOpsTime:
                i += 1
                deviceOp = {}
                cores = set()
                for timeID, _, _, _, core in deviceOpTime["timeseries"]:
                    if "zone_name" in timeID and "FW" in timeID["zone_name"]:
                        if core not in cores:
                            cores.add(core)
                deviceOp["core_usage"] = {"count": len(cores), "cores": [str(core) for core in cores]}
                deviceOp["device_time"] = {
                    analysis: {"series": data["series"], "stats": data["stats"]}
                    for analysis, data in deviceOpTime["analysis"].items()
                }

                if "run_host_id" in timeID:
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
                        if device in devicePreOpTime:
                            rowDict["OP TO OP LATENCY [ns]"] = round(
                                1000 * (analysisData[0]["start_cycle"] - devicePreOpTime[device]) / freq
                            )
                        else:
                            rowDict["OP TO OP LATENCY [ns]"] = 0
                        devicePreOpTime[device] = analysisData[0]["end_cycle"]
                    if analysis == "device_kernel_duration_dm_start":
                        if device in devicePreOpDMStartTime:
                            rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = round(
                                1000 * (analysisData[0]["start_cycle"] - devicePreOpDMStartTime[device]) / freq
                            )
                        else:
                            rowDict["OP TO OP LATENCY BR/NRISC START [ns]"] = 0
                        devicePreOpDMStartTime[device] = analysisData[0]["end_cycle"]

                # Add efficiency metrics if available for this device and operation
                if device in device_efficiency_metrics:
                    from math import nan

                    global_call_count = deviceOp["global_call_count"]
                    trace_id_counter = -1  # Device-only mode doesn't have trace replays
                    lookup_key = (global_call_count, trace_id_counter)
                    metrics = device_efficiency_metrics[device]

                    # Write all metrics to CSV row systematically
                    for base_name, m in metrics.items():
                        is_ipc = "IPC" in base_name
                        suffix = "" if is_ipc else " (%)"
                        # Special handling for SFPU/FPU/MATH "Avg on full grid" legacy names
                        if base_name == "SFPU Util":
                            rowDict["Avg SFPU util on full grid (%)"] = m["avg"].get(lookup_key, nan)
                        elif base_name == "FPU Util":
                            rowDict["Avg FPU util on full grid (%)"] = m["avg"].get(lookup_key, nan)
                        elif base_name == "MATH Util":
                            rowDict["Avg Math util on full grid (%)"] = m["avg"].get(lookup_key, nan)
                        else:
                            rowDict[f"{base_name} Avg{suffix}"] = m["avg"].get(lookup_key, nan)
                        rowDict[f"{base_name} Min{suffix}"] = m["min"].get(lookup_key, nan)
                        rowDict[f"{base_name} Median{suffix}"] = m["median"].get(lookup_key, nan)
                        rowDict[f"{base_name} Max{suffix}"] = m["max"].get(lookup_key, nan)

                rowDicts.append(rowDict)

            def get_core_str_format(core):
                return f"{core[0]}; {core[1]} [ns]"

            allCores = list(deviceData["devices"][device]["cores"])
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

        csv_row_headers = set()
        for row in rowDicts:
            for k in row:
                csv_row_headers.add(k)
        if export_csv:
            with open(allOpsCSVPath, "w") as allOpsCSV:
                allHeaders = []
                for header in OPS_CSV_HEADER + PERF_COUNTER_CSV_HEADERS:
                    if header in csv_row_headers:
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


def generate_reports(
    ops: Dict[int, OpDict],
    deviceOps: DeviceOpsDict,
    traceOps: Dict[int, OpDict],
    signposts: Dict[str, Dict[str, Any]],
    logFolder: Path,
    outputFolder: Optional[Path],
    date: bool,
    nameAppend: Optional[str],
) -> None:
    """Emit the final CSV report plus supporting artifacts."""

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
        csv_rows = []

        prev_device_kernel_end_cycle = {}
        prev_device_dm_start_cycle = {}
        prev_device_fw_end_cycle: Dict[int, int] = {}
        device_ns_per_cycle: Dict[int, Optional[float]] = {}

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
                    padded_logical_field = field + "_PAD[LOGICAL]"
                    headers.append(padded_logical_field)
                    assert field in ioData, "Wrong io tensor shape data format"
                    data[padded_logical_field] = ioData[field]
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
                    assert "device_id" in ioData, "Wrong io tensor memory data format"
                    deviceID = ioData["device_id"]
                    assert "memory_config" in ioData, "Wrong io tensor memory data format"
                    assert "buffer_type" in ioData["memory_config"], "Wrong io tensor memory data format"
                    bufferType = ioData["memory_config"]["buffer_type"].upper()
                    assert "memory_layout" in ioData["memory_config"], "Wrong io tensor memory data format"
                    memoryLayout = ioData["memory_config"]["memory_layout"].upper()
                    data["MEMORY"] = f"DEV_{deviceID}_{bufferType}_{memoryLayout}"

            return headers, data

        def add_io_data(tensors, ioType, target_row):
            ioFields = ["shape", "layout", "dtype", "storage_type"]
            for count, tensor in enumerate(tensors):
                for ioField in ioFields:
                    assert ioField in tensor, "Wrong io tensor fields"
                    ioData = tensor[ioField]
                    fields, data = io_tensor_to_csv(ioField, ioData)
                    for field in fields:
                        header = f"{ioType}_{count}_{field}".upper()
                        target_row[header] = data[field]
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

        timeline_keys = list(ops) + list(traceOps) + list(signposts)
        timeline_keys.sort(key=row_compare)
        childCallKeys = set()
        for row in timeline_keys:
            if type(row) is int:
                if row > ((1 << TRACE_OP_ID_BITSHIFT) - 1):
                    timeline_op_record = traceOps[row]
                else:
                    timeline_op_record = ops[row]
                if "child_calls" in timeline_op_record:
                    for childCall in timeline_op_record["child_calls"]:
                        childCallKeys.add(f"{childCall}_TT_HOST_FUNC [ns]")

        for row in timeline_keys:
            csv_row = {}
            if type(row) is str and "sp" in row:
                headerAndMessage = signposts[row]["data"].split(": ")[-1].split("\n")
                csv_row["OP CODE"] = headerAndMessage[0]
                csv_row["OP TYPE"] = "signpost"
                if len(headerAndMessage) > 1:
                    csv_row["ATTRIBUTES"] = headerAndMessage[1]
                csv_row["HOST START TS"] = int(signposts[row]["tracy_time"])
            elif type(row) is int:
                op = row
                if op > ((1 << TRACE_OP_ID_BITSHIFT) - 1):
                    device_op_record = traceOps[op]
                    device_op_record["global_call_count"] = ((1 << TRACE_OP_ID_BITSHIFT) - 1) & op
                    active_op_record = device_op_record
                else:
                    host_op_record = ops[op]
                    host_op_record["metal_trace_replay_session_id"] = ""
                    if "trac_id" not in host_op_record or host_op_record["metal_trace_id"] is None:
                        host_op_record["metal_trace_id"] = ""
                    active_op_record = host_op_record

                for field, fieldData in active_op_record.items():
                    headerField = csv_header_format(field)
                    # Check if headerField (uppercase) matches any header in OPS_CSV_HEADER (case-insensitive)
                    # If it matches, use the original case from OPS_CSV_HEADER to preserve previous commit's format
                    matching_header = None
                    for ops_header in OPS_CSV_HEADER + PERF_COUNTER_CSV_HEADERS:
                        if headerField == csv_header_format(ops_header):
                            matching_header = ops_header
                            break

                    if matching_header:
                        csv_row[matching_header] = fieldData

                assert "host_time" in active_op_record, "Corrupted op data"
                csv_row["HOST START TS"] = int(active_op_record["host_time"]["ns_since_start"])
                csv_row["HOST END TS"] = int(active_op_record["host_time"]["ns_since_start"]) + int(
                    active_op_record["host_time"]["exec_time_ns"]
                )
                csv_row["HOST DURATION [ns]"] = int(active_op_record["host_time"]["exec_time_ns"])

                if "NOC UTIL (%)" in active_op_record:
                    csv_row["NOC UTIL (%)"] = active_op_record.get("NOC UTIL (%)")
                if "MULTICAST NOC UTIL (%)" in active_op_record:
                    csv_row["MULTICAST NOC UTIL (%)"] = active_op_record.get("MULTICAST NOC UTIL (%)")
                if "DRAM BW UTIL (%)" in active_op_record:
                    csv_row["DRAM BW UTIL (%)"] = active_op_record.get("DRAM BW UTIL (%)")
                if "ETH BW UTIL (%)" in active_op_record:
                    csv_row["ETH BW UTIL (%)"] = active_op_record.get("ETH BW UTIL (%)")
                if "NPE CONG IMPACT (%)" in active_op_record:
                    csv_row["NPE CONG IMPACT (%)"] = active_op_record.get("NPE CONG IMPACT (%)")

                if "kernel_info" in active_op_record:
                    csv_row["COMPUTE KERNEL SOURCE"] = []
                    csv_row["COMPUTE KERNEL HASH"] = []
                    csv_row["DATA MOVEMENT KERNEL SOURCE"] = []
                    csv_row["DATA MOVEMENT KERNEL HASH"] = []
                    for computeKernel in active_op_record["kernel_info"]["compute_kernels"]:
                        csv_row["MATH FIDELITY"] = computeKernel["math_fidelity"]
                        csv_row["COMPUTE KERNEL SOURCE"].append(computeKernel["source"])
                        csv_row["COMPUTE KERNEL HASH"].append(computeKernel["name"])

                    for dmKernel in active_op_record["kernel_info"]["datamovement_kernels"]:
                        csv_row["DATA MOVEMENT KERNEL SOURCE"].append(dmKernel["source"])
                        csv_row["DATA MOVEMENT KERNEL HASH"].append(dmKernel["name"])

                    for kernel, kernelSize in active_op_record["kernel_info"]["kernel_sizes"].items():
                        csv_row[kernel.upper().replace("_", " ") + " [B]"] = kernelSize

                # Extract program hash and cache hit status
                if "op_hash" in active_op_record:
                    csv_row["PROGRAM HASH"] = active_op_record["op_hash"]
                if "program_cache_hit" in active_op_record:
                    csv_row["PROGRAM CACHE HIT"] = active_op_record["program_cache_hit"]

                if "core_usage" in active_op_record:
                    csv_row["CORE COUNT"] = active_op_record["core_usage"]["count"]

                deviceID = active_op_record.get("device_id")
                if deviceID is not None:
                    deviceID = int(deviceID)

                kernel_series = None
                dm_series = None
                kernel_freq = None

                if "device_time" in active_op_record:
                    assert deviceID is not None, "Op has device data without device_id"
                    for analysis, data in active_op_record["device_time"].items():
                        analysisData = data["series"]
                        analysisStats = data["stats"]
                        freq = analysisData[0]["duration_cycles"] / analysisData[0]["duration_ns"]
                        if "per_core" in analysis:
                            assert len(analysisData) >= 1, "Unexpected device data format"
                            headerField = f"{csv_header_format(analysis)} MIN [ns]"
                            csv_row[headerField] = f"{analysisStats['Min'] / freq:.0f}"
                            headerField = f"{csv_header_format(analysis)} MAX [ns]"
                            csv_row[headerField] = f"{analysisStats['Max'] / freq:.0f}"
                            headerField = f"{csv_header_format(analysis)} AVG [ns]"
                            csv_row[headerField] = f"{analysisStats['Average'] / freq:.0f}"
                        else:
                            headerField = f"{csv_header_format(analysis)} [ns]"
                            assert len(analysisData) == 1, "Unexpected device data format"
                            csv_row[headerField] = f"{analysisData[0]['duration_ns']:.0f}"
                        if analysis == "device_fw_duration":
                            csv_row["DEVICE FW START CYCLE"] = analysisData[0]["start_cycle"]
                            csv_row["DEVICE FW END CYCLE"] = analysisData[0]["end_cycle"]
                        if analysis == "device_kernel_duration":
                            kernel_series = analysisData[0]
                            kernel_freq = freq
                        if analysis == "device_kernel_duration_dm_start":
                            dm_series = analysisData[0]
                device_perf_row = active_op_record.pop("_device_perf_row", None)
                if device_perf_row:
                    perf_device_id = device_perf_row.get("DEVICE ID", deviceID)
                    if perf_device_id is None:
                        perf_device_id = deviceID
                    ns_per_cycle = device_ns_per_cycle.get(perf_device_id)
                    if ns_per_cycle is None:
                        ns_per_cycle = compute_ns_per_cycle(device_perf_row)
                        device_ns_per_cycle[perf_device_id] = ns_per_cycle

                    kernel_start_cycle = device_perf_row.get("DEVICE KERNEL START CYCLE")
                    kernel_end_cycle = device_perf_row.get("DEVICE KERNEL END CYCLE")
                    dm_start_cycle = device_perf_row.get("DEVICE KERNEL DM START CYCLE")
                    dm_end_cycle = device_perf_row.get("DEVICE KERNEL DM END CYCLE")

                    # Prefer the C++-computed op-to-op latency if it is present in the perf row.
                    # The Python recomputation uses host ordering which can diverge from device ordering
                    # under async/multi-device execution; using the authoritative device-side value keeps
                    # python and cpp reports consistent.
                    perf_kernel_latency = device_perf_row.get("OP TO OP LATENCY [ns]")
                    if perf_kernel_latency not in (None, ""):
                        csv_row["OP TO OP LATENCY [ns]"] = perf_kernel_latency
                    elif (
                        ns_per_cycle
                        and kernel_start_cycle is not None
                        and kernel_end_cycle is not None
                        and perf_device_id is not None
                    ):
                        if perf_device_id in prev_device_kernel_end_cycle:
                            csv_row["OP TO OP LATENCY [ns]"] = round(
                                (kernel_start_cycle - prev_device_kernel_end_cycle[perf_device_id]) * ns_per_cycle
                            )
                        else:
                            csv_row["OP TO OP LATENCY [ns]"] = 0

                    # Track end cycle for fallback computation.
                    if perf_device_id is not None and kernel_end_cycle is not None:
                        prev_device_kernel_end_cycle[perf_device_id] = kernel_end_cycle

                    perf_dm_latency = device_perf_row.get("OP TO OP LATENCY BR/NRISC START [ns]")
                    if perf_dm_latency not in (None, ""):
                        csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = perf_dm_latency
                    elif (
                        ns_per_cycle
                        and dm_start_cycle is not None
                        and dm_end_cycle is not None
                        and perf_device_id is not None
                    ):
                        if perf_device_id in prev_device_dm_start_cycle:
                            csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = round(
                                (dm_start_cycle - prev_device_dm_start_cycle[perf_device_id]) * ns_per_cycle
                            )
                        else:
                            csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = 0

                    # Track end cycle for fallback computation.
                    if perf_device_id is not None and dm_end_cycle is not None:
                        prev_device_dm_start_cycle[perf_device_id] = dm_end_cycle

                    if "OP TO OP LATENCY [ns]" not in csv_row and perf_device_id is not None:
                        csv_row["OP TO OP LATENCY [ns]"] = 0
                    if "OP TO OP LATENCY BR/NRISC START [ns]" not in csv_row and perf_device_id is not None:
                        csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = 0

                    skip_headers = {
                        "GLOBAL CALL COUNT",
                        "DEVICE ID",
                        "CORE COUNT",
                        "METAL TRACE ID",
                        "METAL TRACE REPLAY SESSION ID",
                        "OP TO OP LATENCY [ns]",
                        "OP TO OP LATENCY BR/NRISC START [ns]",
                    }
                    for header, value in device_perf_row.items():
                        if header in skip_headers:
                            continue
                        if header not in OPS_CSV_HEADER and header not in _PERF_COUNTER_CSV_HEADERS_SET:
                            continue
                        if value in (None, ""):
                            continue
                        if header not in csv_row or csv_row[header] == "":
                            csv_row[header] = value

                if kernel_series and kernel_freq and deviceID is not None and "OP TO OP LATENCY [ns]" not in csv_row:
                    if deviceID in prev_device_kernel_end_cycle:
                        csv_row["OP TO OP LATENCY [ns]"] = round(
                            (kernel_series["start_cycle"] - prev_device_kernel_end_cycle[deviceID]) / kernel_freq
                        )
                    else:
                        csv_row["OP TO OP LATENCY [ns]"] = 0
                    prev_device_kernel_end_cycle[deviceID] = kernel_series["end_cycle"]

                if (
                    dm_series
                    and kernel_freq
                    and deviceID is not None
                    and "OP TO OP LATENCY BR/NRISC START [ns]" not in csv_row
                ):
                    if deviceID in prev_device_dm_start_cycle:
                        csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = round(
                            (dm_series["start_cycle"] - prev_device_dm_start_cycle[deviceID]) / kernel_freq
                        )
                    else:
                        csv_row["OP TO OP LATENCY BR/NRISC START [ns]"] = 0
                    prev_device_dm_start_cycle[deviceID] = dm_series["end_cycle"]

                # Convert avg counter values to percentages for "Avg ... util on full grid (%)" columns
                kernel_duration_cycles = None
                if kernel_series:
                    kernel_duration_cycles = kernel_series.get("duration_cycles")
                elif device_perf_row:
                    kernel_start_cycle = device_perf_row.get("DEVICE KERNEL START CYCLE")
                    kernel_end_cycle = device_perf_row.get("DEVICE KERNEL END CYCLE")
                    if kernel_start_cycle is not None and kernel_end_cycle is not None:
                        kernel_duration_cycles = kernel_end_cycle - kernel_start_cycle

                if kernel_duration_cycles is not None and kernel_duration_cycles > 0:
                    if "avg_sfpu_count" in active_op_record:
                        avg_sfpu_val = active_op_record.get("avg_sfpu_count")
                        if avg_sfpu_val is not None and not isnan(avg_sfpu_val):
                            csv_row["Avg SFPU util on full grid (%)"] = avg_sfpu_val / kernel_duration_cycles * 100
                    if "avg_fpu_count" in active_op_record:
                        avg_fpu_val = active_op_record.get("avg_fpu_count")
                        if avg_fpu_val is not None and not isnan(avg_fpu_val):
                            csv_row["Avg FPU util on full grid (%)"] = avg_fpu_val / kernel_duration_cycles * 100
                    if "avg_math_count" in active_op_record:
                        avg_math_val = active_op_record.get("avg_math_count")
                        if avg_math_val is not None and not isnan(avg_math_val):
                            csv_row["Avg Math util on full grid (%)"] = avg_math_val / kernel_duration_cycles * 100

                if "child_calls" in active_op_record:
                    for childCall, duration in active_op_record["child_calls"].items():
                        headerField = f"{childCall}_TT_HOST_FUNC [ns]"
                        csv_row[headerField] = f"{duration:.0f}"

                assert "input_tensors" in active_op_record, "Ops must have input tensors"
                if "optional_input_tensors" in active_op_record:
                    add_io_data(
                        active_op_record["input_tensors"] + active_op_record["optional_input_tensors"],
                        "INPUT",
                        csv_row,
                    )
                else:
                    add_io_data(active_op_record["input_tensors"], "INPUT", csv_row)

                if "output_tensors" in active_op_record:
                    add_io_data(active_op_record["output_tensors"], "OUTPUT", csv_row)

                if "performance_model" in active_op_record:
                    csv_row["PM IDEAL [ns]"] = active_op_record["performance_model"]["ideal_ns"]
                    csv_row["PM COMPUTE [ns]"] = active_op_record["performance_model"]["compute_ns"]
                    csv_row["PM BANDWIDTH [ns]"] = active_op_record["performance_model"]["bandwidth_ns"]
                    csv_row["PM REQ I BW"] = active_op_record["performance_model"]["input_bws"]
                    csv_row["PM REQ O BW"] = active_op_record["performance_model"]["output_bws"]

                    if "DEVICE KERNEL DURATION [ns]" in csv_row:
                        try:
                            fpu_util = (
                                100.0
                                * float(csv_row["PM COMPUTE [ns]"])
                                / float(csv_row["DEVICE KERNEL DURATION [ns]"])
                            )
                            csv_row["PM FPU UTIL (%)"] = round(fpu_util, 3)
                        except ZeroDivisionError:
                            csv_row["PM FPU UTIL (%)"] = 0.0

            csv_rows.append(csv_row)

        # Determine which perf counter headers have data in any row
        all_row_keys = set()
        for row in csv_rows:
            all_row_keys.update(row.keys())
        active_perf_headers = [h for h in PERF_COUNTER_CSV_HEADERS if h in all_row_keys]

        ioHeaderIndex = OPS_CSV_HEADER.index("INPUTS")
        allHeaders = (
            OPS_CSV_HEADER[:ioHeaderIndex]
            + tensorCSVData["INPUT"]["headers"]
            + tensorCSVData["OUTPUT"]["headers"]
            + OPS_CSV_HEADER[ioHeaderIndex + 2 :]
            + active_perf_headers
            + sorted(list(childCallKeys))
        )
        writer = csv.DictWriter(allOpsCSV, fieldnames=allHeaders)
        writer.writeheader()
        for csv_row in csv_rows:
            for field, fieldData in csv_row.items():
                csv_row[field] = str(fieldData).replace(",", ";")
            writer.writerow(csv_row)
    logger.info(f"OPs csv generated at: {allOpsCSVPath}")


def analyzeNoCTraces(logFolder: Path):
    """Attempts to import tt-npe from $PYTHONPATH and process noc traces to
    obtain per-operation DRAM BW and NoC utilization statistics and create
    visualizer timeline files"""
    try:
        from npe_analyze_noc_trace_dir import analyze_noc_traces_in_dir

        logger.info(f"tt-npe module imported successfully; analyzing noc traces ... ")
        return analyze_noc_traces_in_dir(
            noc_trace_dir=logFolder,
            emit_viz_timeline_files=True,
            quiet=True,
            compress_timeline_files=True,
        )
    except ImportError:
        logger.warning("Could not import tt-npe module. Ensure tt-npe is built, then source 'tt-npe/ENV_SETUP'")
        return None
    except Exception as e:
        logger.error("Unexpected error occurred when analyzing noc traces, aborting ... ")
        logger.error(" ↳ " + repr(e))
        return None


def process_ops(
    output_folder: Optional[Path],
    name_append: Optional[str],
    date: bool,
    device_only: bool = False,
    analyze_noc_traces: bool = False,
    device_analysis_types: Tuple[str, ...] | List[str] = (),
    force_legacy_device_logs: bool = False,
) -> None:
    """Top-level entry point used by both CLI and importers."""

    if not output_folder:
        output_folder = PROFILER_ARTIFACTS_DIR
    logFolder = generate_logs_folder(output_folder)
    reportFolder = generate_reports_folder(output_folder)

    ops, signposts, traceReplays = import_tracy_op_logs(logFolder)

    if ops and not device_only:
        deviceOps, traceOps = append_device_data(
            ops,
            traceReplays,
            logFolder,
            analyze_noc_traces,
            device_analysis_types,
            force_legacy_device_logs=force_legacy_device_logs,
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
@click.option(
    "--force-legacy-device-logs",
    is_flag=True,
    help="Force use of legacy device log parsing instead of cpp_device_perf_report.csv.",
)
def main(
    output_folder, name_append, date, device_only, analyze_noc_traces, device_analysis_types, force_legacy_device_logs
):
    if output_folder:
        output_folder = Path(output_folder)
    process_ops(
        output_folder,
        name_append,
        date,
        device_only,
        analyze_noc_traces,
        device_analysis_types,
        force_legacy_device_logs=force_legacy_device_logs,
    )


if __name__ == "__main__":
    main()
