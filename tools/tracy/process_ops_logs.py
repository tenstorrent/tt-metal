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
    "DRAM BW UTIL (%)",
    "NPE CONG IMPACT (%)",
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
]


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


def load_device_perf_report(report_path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Parse cpp_device_perf_report.csv into a per-device/per-op mapping."""

    per_device: Dict[int, Dict[int, Dict[str, Any]]] = {}
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
            if device_id is None or op_id is None:
                continue

            if device_id not in per_device:
                per_device[device_id] = {}
            per_device[device_id][op_id] = parsed_row

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
                        opData = json.loads(jsonStr)
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
                        assert len(opDataList) > 3, "Wrong cached op info format"
                        opHash = int(opDataList[1])
                        deviceID = int(opDataList[2])
                        opID = int(opDataList[3])
                        assert deviceID in cached_ops, "Expected hashed op info is not found"
                        assert opHash in cached_ops[deviceID], "Expected hashed op info is not found"
                        opData = cached_ops[deviceID][opHash].copy()
                        opData["global_call_count"] = opID
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

    tracyOpTimesData = []
    with open(tracyOpTimesLog, "r") as csvFile:
        csvReader = csv.DictReader(csvFile)
        tracyOpTimesData = list(csvReader)

    for op in tracyOpTimesData:
        if "TT_DNN" in op["name"] or "TT_METAL" in op["name"]:
            opID = int(op["zone_text"].split(":")[-1])
            assert opID in ops, f"Op time for op {opID} must present"
            ops[opID]["host_time"] = op

    for op in tracyOpTimesData:
        if op["special_parent_text"] and "id:" in op["special_parent_text"]:
            parentOpID = int(op["special_parent_text"].split(":")[-1])

            if "child_calls" in ops[parentOpID]:
                if op["name"] in ops[parentOpID]["child_calls"]:
                    ops[parentOpID]["child_calls"][op["name"]] += int(op["exec_time_ns"])
                else:
                    ops[parentOpID]["child_calls"][op["name"]] = int(op["exec_time_ns"])
            else:
                ops[parentOpID]["child_calls"] = {op["name"]: int(op["exec_time_ns"])}

    return ops, signposts, traceReplays


def host_device_op_compare(op: OpDict) -> Tuple[int, int]:
    """Comparison key that keeps ops ordered by host id, then replay session."""

    if "metal_trace_replay_session_id" in op:
        return int(op["global_call_count"]), int(op["metal_trace_replay_session_id"])
    else:
        return int(op["global_call_count"]), 0


# Generate a map of OP reference list per device.
def get_device_op_data(ops: Dict[int, OpDict]) -> Tuple[DeviceOpsDict, bool]:
    """Group host ops per device and record whether trace runs exist."""

    logger.info(f"Getting device ops")
    deviceOps = {}
    hasTraceRuns = False
    for opID, opData in ops.items():
        if "device_id" in opData:
            deviceID = opData["device_id"]
            if deviceID not in deviceOps:
                deviceOps[deviceID] = [opData]
            else:
                deviceOps[deviceID].append(opData)
        if "metal_trace_id" in opData and opData["metal_trace_id"] is not None:
            hasTraceRuns = True

    for deviceID in deviceOps:
        deviceOps[deviceID].sort(key=host_device_op_compare)

    return deviceOps, hasTraceRuns


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


def _load_legacy_device_ops(
    log_folder: Path, device_analysis_types: Tuple[str, ...] | List[str]
) -> Dict[int, Dict[int, OpDict]]:
    """Parse device-side logs via import_log_run_stats for legacy enrichment."""

    device_log_path = Path(log_folder) / PROFILER_DEVICE_SIDE_LOG
    if not device_log_path.is_file():
        raise AssertionError(
            f"{PROFILER_CPP_DEVICE_PERF_REPORT} not found and legacy device log "
            f"{PROFILER_DEVICE_SIDE_LOG} is also missing in {log_folder}."
        )

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

    per_device_ops: Dict[int, Dict[int, OpDict]] = defaultdict(dict)
    for device_id, device_info in device_data["devices"].items():
        tensix_ops = device_info["cores"]["DEVICE"]["riscs"]["TENSIX"]["ops"]
        for device_op_time in tensix_ops:
            device_op = _convert_device_op_entry(device_op_time, freq)
            per_device_ops[device_id][device_op["global_call_count"]] = device_op
    return per_device_ops


def _enrich_ops_from_perf_csv(
    host_ops_by_device: DeviceOpsDict,
    device_perf_by_device: Dict[int, Dict[int, Dict[str, Any]]],
    trace_replays: Optional[TraceReplayDict],
) -> DeviceOpsDict:
    for device_id in host_ops_by_device:
        assert (
            device_id in device_perf_by_device
        ), f"Device {device_id} present in host logs but missing from {PROFILER_CPP_DEVICE_PERF_REPORT}"

        enriched_ops = []
        for host_op in host_ops_by_device[device_id]:
            op_id = int(host_op["global_call_count"])
            assert (
                op_id in device_perf_by_device[device_id]
            ), f"Device data missing: Op {op_id} not present in {PROFILER_CPP_DEVICE_PERF_REPORT} for device {device_id}"

            perf_row = device_perf_by_device[device_id][op_id].copy()
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
) -> DeviceOpsDict:
    legacy_device_ops = _load_legacy_device_ops(log_folder, device_analysis_types)

    for device_id, host_ops in host_ops_by_device.items():
        device_entries = legacy_device_ops.get(device_id, {})
        enriched_ops = []
        for host_op in host_ops:
            op_id = int(host_op["global_call_count"])
            enriched_op = copy.deepcopy(host_op)
            device_entry = device_entries.get(op_id)
            if device_entry:
                enriched_op["core_usage"] = device_entry.get("core_usage")
                enriched_op["device_time"] = device_entry.get("device_time")
            enriched_ops.append(enriched_op)
        host_ops_by_device[device_id] = enriched_ops
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

    host_ops_by_device, _ = get_device_op_data(ops)
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
        host_ops_by_device = _enrich_ops_from_device_logs(host_ops_by_device, logFolder, device_analysis_types)

    trace_ops_by_augmented_id = _build_trace_ops_mapping(host_ops_by_device, ops)

    if analyze_noc_traces:
        npe_stats = analyzeNoCTraces(logFolder)
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
                for header in OPS_CSV_HEADER:
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
                    if headerField in OPS_CSV_HEADER:
                        csv_row[headerField] = fieldData

                assert "host_time" in active_op_record, "Corrupted op data"
                csv_row["HOST START TS"] = int(active_op_record["host_time"]["ns_since_start"])
                csv_row["HOST END TS"] = int(active_op_record["host_time"]["ns_since_start"]) + int(
                    active_op_record["host_time"]["exec_time_ns"]
                )
                csv_row["HOST DURATION [ns]"] = int(active_op_record["host_time"]["exec_time_ns"])

                if "NOC UTIL (%)" in active_op_record:
                    csv_row["NOC UTIL (%)"] = active_op_record.get("NOC UTIL (%)")
                if "DRAM BW UTIL (%)" in active_op_record:
                    csv_row["DRAM BW UTIL (%)"] = active_op_record.get("DRAM BW UTIL (%)")
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

                    if (
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
                        prev_device_kernel_end_cycle[perf_device_id] = kernel_end_cycle

                    if (
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
                        if header not in OPS_CSV_HEADER:
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
def main(output_folder, name_append, date, device_only, analyze_noc_traces, device_analysis_types, force_legacy_device_logs):
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
