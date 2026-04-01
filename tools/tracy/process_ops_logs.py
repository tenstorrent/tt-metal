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
    # L1 Bank 0 Port 1: arch-specific (WH: Unpacker#1/ECC/Pack1, BH: Unified Packer)
    "L1 Packer Port Util Min (%)",
    "L1 Packer Port Util Median (%)",
    "L1 Packer Port Util Max (%)",
    "L1 Packer Port Util Avg (%)",
    # L1 back-pressure metrics: (req - grant) / req * 100
    "NOC Ring 0 Outgoing Backpressure Min (%)",
    "NOC Ring 0 Outgoing Backpressure Median (%)",
    "NOC Ring 0 Outgoing Backpressure Max (%)",
    "NOC Ring 0 Outgoing Backpressure Avg (%)",
    "NOC Ring 0 Incoming Backpressure Min (%)",
    "NOC Ring 0 Incoming Backpressure Median (%)",
    "NOC Ring 0 Incoming Backpressure Max (%)",
    "NOC Ring 0 Incoming Backpressure Avg (%)",
    "NOC Ring 1 Outgoing Backpressure Min (%)",
    "NOC Ring 1 Outgoing Backpressure Median (%)",
    "NOC Ring 1 Outgoing Backpressure Max (%)",
    "NOC Ring 1 Outgoing Backpressure Avg (%)",
    "NOC Ring 1 Incoming Backpressure Min (%)",
    "NOC Ring 1 Incoming Backpressure Median (%)",
    "NOC Ring 1 Incoming Backpressure Max (%)",
    "NOC Ring 1 Incoming Backpressure Avg (%)",
    "L1 Unpacker Backpressure Min (%)",
    "L1 Unpacker Backpressure Median (%)",
    "L1 Unpacker Backpressure Max (%)",
    "L1 Unpacker Backpressure Avg (%)",
    "L1 Packer Port Backpressure Min (%)",
    "L1 Packer Port Backpressure Median (%)",
    "L1 Packer Port Backpressure Max (%)",
    "L1 Packer Port Backpressure Avg (%)",
    # Fidelity cycle breakdown
    "HiFi2 Instrn Rate Min (%)",
    "HiFi2 Instrn Rate Median (%)",
    "HiFi2 Instrn Rate Max (%)",
    "HiFi2 Instrn Rate Avg (%)",
    "LoFi Instrn Rate Min (%)",
    "LoFi Instrn Rate Median (%)",
    "LoFi Instrn Rate Max (%)",
    "LoFi Instrn Rate Avg (%)",
    # Math pipeline stall breakdown
    "Math Src Data Ready Rate Min (%)",
    "Math Src Data Ready Rate Median (%)",
    "Math Src Data Ready Rate Max (%)",
    "Math Src Data Ready Rate Avg (%)",
    "SrcA Write Port Blocked Rate Min (%)",
    "SrcA Write Port Blocked Rate Median (%)",
    "SrcA Write Port Blocked Rate Max (%)",
    "SrcA Write Port Blocked Rate Avg (%)",
    "Dest Read Backpressure Min (%)",
    "Dest Read Backpressure Median (%)",
    "Dest Read Backpressure Max (%)",
    "Dest Read Backpressure Avg (%)",
    "Math Dest Write Port Stall Rate Min (%)",
    "Math Dest Write Port Stall Rate Median (%)",
    "Math Dest Write Port Stall Rate Max (%)",
    "Math Dest Write Port Stall Rate Avg (%)",
    "Math Scoreboard Stall Rate Min (%)",
    "Math Scoreboard Stall Rate Median (%)",
    "Math Scoreboard Stall Rate Max (%)",
    "Math Scoreboard Stall Rate Avg (%)",
    # Instruction issue rates (per cycle, not %)
    "Unpack Instrn Issue Rate T0 Min",
    "Unpack Instrn Issue Rate T0 Median",
    "Unpack Instrn Issue Rate T0 Max",
    "Unpack Instrn Issue Rate T0 Avg",
    "Math Instrn Issue Rate T1 Min",
    "Math Instrn Issue Rate T1 Median",
    "Math Instrn Issue Rate T1 Max",
    "Math Instrn Issue Rate T1 Avg",
    "Pack Instrn Issue Rate T2 Min",
    "Pack Instrn Issue Rate T2 Median",
    "Pack Instrn Issue Rate T2 Max",
    "Pack Instrn Issue Rate T2 Avg",
    # === NEW: Per-type instruction issue efficiency ===
    "CFG Instrn Avail Rate T0 Min (%)",
    "CFG Instrn Avail Rate T0 Median (%)",
    "CFG Instrn Avail Rate T0 Max (%)",
    "CFG Instrn Avail Rate T0 Avg (%)",
    "SYNC Instrn Avail Rate T0 Min (%)",
    "SYNC Instrn Avail Rate T0 Median (%)",
    "SYNC Instrn Avail Rate T0 Max (%)",
    "SYNC Instrn Avail Rate T0 Avg (%)",
    "THCON Instrn Avail Rate T0 Min (%)",
    "THCON Instrn Avail Rate T0 Median (%)",
    "THCON Instrn Avail Rate T0 Max (%)",
    "THCON Instrn Avail Rate T0 Avg (%)",
    "MOVE Instrn Avail Rate T0 Min (%)",
    "MOVE Instrn Avail Rate T0 Median (%)",
    "MOVE Instrn Avail Rate T0 Max (%)",
    "MOVE Instrn Avail Rate T0 Avg (%)",
    "MATH Instrn Avail Rate T1 Min (%)",
    "MATH Instrn Avail Rate T1 Median (%)",
    "MATH Instrn Avail Rate T1 Max (%)",
    "MATH Instrn Avail Rate T1 Avg (%)",
    "UNPACK Instrn Avail Rate T0 Min (%)",
    "UNPACK Instrn Avail Rate T0 Median (%)",
    "UNPACK Instrn Avail Rate T0 Max (%)",
    "UNPACK Instrn Avail Rate T0 Avg (%)",
    "PACK Instrn Avail Rate T2 Min (%)",
    "PACK Instrn Avail Rate T2 Median (%)",
    "PACK Instrn Avail Rate T2 Max (%)",
    "PACK Instrn Avail Rate T2 Avg (%)",
    # === NEW: Stall breakdown (% of total stalls per thread) ===
    "THCON Idle Stall Pct T0 Min (%)",
    "THCON Idle Stall Pct T0 Median (%)",
    "THCON Idle Stall Pct T0 Max (%)",
    "THCON Idle Stall Pct T0 Avg (%)",
    "MOVE Idle Stall Pct T0 Min (%)",
    "MOVE Idle Stall Pct T0 Median (%)",
    "MOVE Idle Stall Pct T0 Max (%)",
    "MOVE Idle Stall Pct T0 Avg (%)",
    "MMIO Idle Stall Pct T1 Min (%)",
    "MMIO Idle Stall Pct T1 Median (%)",
    "MMIO Idle Stall Pct T1 Max (%)",
    "MMIO Idle Stall Pct T1 Avg (%)",
    "SFPU Idle Stall Pct T1 Min (%)",
    "SFPU Idle Stall Pct T1 Median (%)",
    "SFPU Idle Stall Pct T1 Max (%)",
    "SFPU Idle Stall Pct T1 Avg (%)",
    # === NEW: Write port blocking ===
    "SrcB Write Port Blocked Rate Min (%)",
    "SrcB Write Port Blocked Rate Median (%)",
    "SrcB Write Port Blocked Rate Max (%)",
    "SrcB Write Port Blocked Rate Avg (%)",
    "SrcA Write Actual Efficiency Min (%)",
    "SrcA Write Actual Efficiency Median (%)",
    "SrcA Write Actual Efficiency Max (%)",
    "SrcA Write Actual Efficiency Avg (%)",
    "SrcB Write Actual Efficiency Min (%)",
    "SrcB Write Actual Efficiency Median (%)",
    "SrcB Write Actual Efficiency Max (%)",
    "SrcB Write Actual Efficiency Avg (%)",
    # === NEW: Fidelity analysis ===
    "HiFi4 Instrn Rate Min (%)",
    "HiFi4 Instrn Rate Median (%)",
    "HiFi4 Instrn Rate Max (%)",
    "HiFi4 Instrn Rate Avg (%)",
    "Fidelity Phase Overhead Min (%)",
    "Fidelity Phase Overhead Median (%)",
    "Fidelity Phase Overhead Max (%)",
    "Fidelity Phase Overhead Avg (%)",
    # === NEW: Packer engine granularity (WH) ===
    "Packer Engine 0 Util Min (%)",
    "Packer Engine 0 Util Median (%)",
    "Packer Engine 0 Util Max (%)",
    "Packer Engine 0 Util Avg (%)",
    "Packer Engine 1 Util Min (%)",
    "Packer Engine 1 Util Median (%)",
    "Packer Engine 1 Util Max (%)",
    "Packer Engine 1 Util Avg (%)",
    "Packer Engine 2 Util Min (%)",
    "Packer Engine 2 Util Median (%)",
    "Packer Engine 2 Util Max (%)",
    "Packer Engine 2 Util Avg (%)",
    # === NEW: Low priority waits ===
    "MMIO Idle Wait T0 Min (%)",
    "MMIO Idle Wait T0 Median (%)",
    "MMIO Idle Wait T0 Max (%)",
    "MMIO Idle Wait T0 Avg (%)",
    "SFPU Idle Wait T1 Min (%)",
    "SFPU Idle Wait T1 Median (%)",
    "SFPU Idle Wait T1 Max (%)",
    "SFPU Idle Wait T1 Avg (%)",
    "THCON Idle Wait T0 Min (%)",
    "THCON Idle Wait T0 Median (%)",
    "THCON Idle Wait T0 Max (%)",
    "THCON Idle Wait T0 Avg (%)",
    "MOVE Idle Wait T0 Min (%)",
    "MOVE Idle Wait T0 Median (%)",
    "MOVE Idle Wait T0 Max (%)",
    "MOVE Idle Wait T0 Avg (%)",
    # === NEW: RISC Core L1 util (BH only) ===
    "RISC Core L1 Util Min (%)",
    "RISC Core L1 Util Median (%)",
    "RISC Core L1 Util Max (%)",
    "RISC Core L1 Util Avg (%)",
    # === L1 composite metrics ===
    "L1 Total Bandwidth Util Min (%)",
    "L1 Total Bandwidth Util Median (%)",
    "L1 Total Bandwidth Util Max (%)",
    "L1 Total Bandwidth Util Avg (%)",
    "L1 Read vs Write Ratio Min (%)",
    "L1 Read vs Write Ratio Median (%)",
    "L1 Read vs Write Ratio Max (%)",
    "L1 Read vs Write Ratio Avg (%)",
    "NOC Ring 0 Asymmetry Min (%)",
    "NOC Ring 0 Asymmetry Median (%)",
    "NOC Ring 0 Asymmetry Max (%)",
    "NOC Ring 0 Asymmetry Avg (%)",
    "L1 Contention Index Min (%)",
    "L1 Contention Index Median (%)",
    "L1 Contention Index Max (%)",
    "L1 Contention Index Avg (%)",
    "Unpacker L1 Efficiency Min (%)",
    "Unpacker L1 Efficiency Median (%)",
    "Unpacker L1 Efficiency Max (%)",
    "Unpacker L1 Efficiency Avg (%)",
    "Packer L1 Efficiency Min (%)",
    "Packer L1 Efficiency Median (%)",
    "Packer L1 Efficiency Max (%)",
    "Packer L1 Efficiency Avg (%)",
    "NOC vs Compute Balance Min (%)",
    "NOC vs Compute Balance Median (%)",
    "NOC vs Compute Balance Max (%)",
    "NOC vs Compute Balance Avg (%)",
    "TDMA vs NOC L1 Share Min (%)",
    "TDMA vs NOC L1 Share Median (%)",
    "TDMA vs NOC L1 Share Max (%)",
    "TDMA vs NOC L1 Share Avg (%)",
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

            def compute_ratio_metric(numerator_name, denominator_name, scale=100):
                """Compute numerator / denominator * scale per core, aggregate by op."""
                num = get_counter_series(numerator_name)
                den = get_counter_series(denominator_name)
                ratio = (num / den * scale).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                return {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            def compute_complement_metric(counter_name, total_name):
                """Compute (total - counter) / total * 100 — for blocked/stall rates."""
                val = get_counter_series(counter_name)
                total = get_counter_series(total_name)
                ratio = ((total - val) / total * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                return {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            def compute_backpressure(req_ch0, req_ch1, grant_ch0, grant_ch1):
                """Compute avg back-pressure across two channels: (req-grant)/req * 100."""
                r0 = get_counter_series(req_ch0)
                r1 = get_counter_series(req_ch1)
                g0 = get_counter_series(grant_ch0)
                g1 = get_counter_series(grant_ch1)
                bp = (((r0 - g0) + (r1 - g1)) / (r0 + r1) * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = bp.groupby(level=["run_host_id", "trace_id_count"])
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
            math_pack_eff = (available_math / packer_busy * 100).clip(upper=100).replace([float("inf"), -float("inf")], nan)
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

            # L1 Port 1 (arch-specific: BH unified packer or WH unpacker#1/ECC/pack1)
            l1_packer_port_util = {}
            if has_counter("L1_0_UNIFIED_PACKER"):
                l1_packer_port_util = compute_util_metric("L1_0_UNIFIED_PACKER")
            elif has_counter("L1_0_UNPACKER_1_ECC_PACK1"):
                l1_packer_port_util = compute_util_metric("L1_0_UNPACKER_1_ECC_PACK1")

            # L1 back-pressure metrics (from grant counters)
            noc_r0_out_bp = {}
            noc_r0_in_bp = {}
            if has_counter("L1_0_NOC_RING0_OUTGOING_0") and has_counter("L1_0_NOC_RING0_OUTGOING_0_GRANT"):
                noc_r0_out_bp = compute_backpressure(
                    "L1_0_NOC_RING0_OUTGOING_0",
                    "L1_0_NOC_RING0_OUTGOING_1",
                    "L1_0_NOC_RING0_OUTGOING_0_GRANT",
                    "L1_0_NOC_RING0_OUTGOING_1_GRANT",
                )
            if has_counter("L1_0_NOC_RING0_INCOMING_0") and has_counter("L1_0_NOC_RING0_INCOMING_0_GRANT"):
                noc_r0_in_bp = compute_backpressure(
                    "L1_0_NOC_RING0_INCOMING_0",
                    "L1_0_NOC_RING0_INCOMING_1",
                    "L1_0_NOC_RING0_INCOMING_0_GRANT",
                    "L1_0_NOC_RING0_INCOMING_1_GRANT",
                )

            # === Grant counter derived metrics ===
            # Fidelity cycle breakdown
            hifi2_rate = {}
            lofi_rate = {}
            if has_counter("INSTRN_2_HF_CYCLES") and has_counter("MATH_INSTRN_STARTED"):
                num = get_counter_series("INSTRN_2_HF_CYCLES")
                den = get_counter_series("MATH_INSTRN_STARTED")
                ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                hifi2_rate = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }
            if has_counter("INSTRN_1_HF_CYCLE") and has_counter("MATH_INSTRN_STARTED"):
                num = get_counter_series("INSTRN_1_HF_CYCLE")
                den = get_counter_series("MATH_INSTRN_STARTED")
                ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                lofi_rate = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # Math source data readiness
            math_src_ready = {}
            if has_counter("MATH_INSTRN_NOT_BLOCKED_SRC") and has_counter("MATH_INSTRN_AVAILABLE"):
                num = get_counter_series("MATH_INSTRN_NOT_BLOCKED_SRC")
                den = get_counter_series("MATH_INSTRN_AVAILABLE")
                ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                math_src_ready = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # SrcA write port blocked rate
            srca_blocked = {}
            if has_counter("SRCA_WRITE_AVAILABLE") and has_counter("SRCA_WRITE_NOT_BLOCKED_OVR"):
                avail = get_counter_series("SRCA_WRITE_AVAILABLE")
                unblocked = get_counter_series("SRCA_WRITE_NOT_BLOCKED_OVR")
                ratio = ((avail - unblocked) / avail * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                srca_blocked = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # Dest read backpressure
            dest_bp = {}
            if has_counter("PACKER_DEST_READ_AVAILABLE") and has_counter("DEST_READ_GRANTED_0"):
                req = get_counter_series("PACKER_DEST_READ_AVAILABLE")
                grant = get_counter_series("DEST_READ_GRANTED_0")
                ratio = ((req - grant) / req * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                dest_bp = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # Math dest write port stall and scoreboard stall
            math_dest_wr_stall = {}
            math_scoreboard_stall = {}
            if has_counter("MATH_INSTRN_AVAILABLE") and has_counter("MATH_NOT_STALLED_DEST_WR_PORT"):
                avail = get_counter_series("MATH_INSTRN_AVAILABLE")
                unstalled = get_counter_series("MATH_NOT_STALLED_DEST_WR_PORT")
                ratio = ((avail - unstalled) / avail * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                math_dest_wr_stall = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }
            if has_counter("MATH_INSTRN_AVAILABLE") and has_counter("AVAILABLE_MATH"):
                avail = get_counter_series("MATH_INSTRN_AVAILABLE")
                unstalled = get_counter_series("AVAILABLE_MATH")
                ratio = ((avail - unstalled) / avail * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                math_scoreboard_stall = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # Instruction issue rates per thread
            unpack_issue_rate = {}
            math_issue_rate = {}
            pack_issue_rate = {}
            if has_counter("UNPACK_INSTRN_ISSUED_0"):
                unpack_issue_rate = compute_util_metric("UNPACK_INSTRN_ISSUED_0", scale=1)
            if has_counter("FPU_INSTRN_ISSUED_1"):
                math_issue_rate = compute_util_metric("FPU_INSTRN_ISSUED_1", scale=1)
            if has_counter("PACK_INSTRN_ISSUED_2"):
                pack_issue_rate = compute_util_metric("PACK_INSTRN_ISSUED_2", scale=1)

            # === New metrics: L1 Bank 1 ===
            noc_r1_out_util = {}
            noc_r1_in_util = {}
            if has_counter("L1_1_NOC_RING1_OUTGOING_0") and has_counter("L1_1_NOC_RING1_OUTGOING_1"):
                noc_r1_out_util = compute_avg_channel_util("L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1")
            if has_counter("L1_1_NOC_RING1_INCOMING_0") and has_counter("L1_1_NOC_RING1_INCOMING_1"):
                noc_r1_in_util = compute_avg_channel_util("L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1")

            # === Derived stall metrics (req - grant) / req * 100 ===
            # These are equivalent to the BH hardware stall_cnt but computed in software.
            noc_r1_out_bp = {}
            noc_r1_in_bp = {}
            l1_unpacker_bp = {}
            l1_packer_port_bp = {}
            if has_counter("L1_1_NOC_RING1_OUTGOING_0") and has_counter("L1_1_NOC_RING1_OUTGOING_0_GRANT"):
                noc_r1_out_bp = compute_backpressure(
                    "L1_1_NOC_RING1_OUTGOING_0",
                    "L1_1_NOC_RING1_OUTGOING_1",
                    "L1_1_NOC_RING1_OUTGOING_0_GRANT",
                    "L1_1_NOC_RING1_OUTGOING_1_GRANT",
                )
            if has_counter("L1_1_NOC_RING1_INCOMING_0") and has_counter("L1_1_NOC_RING1_INCOMING_0_GRANT"):
                noc_r1_in_bp = compute_backpressure(
                    "L1_1_NOC_RING1_INCOMING_0",
                    "L1_1_NOC_RING1_INCOMING_1",
                    "L1_1_NOC_RING1_INCOMING_0_GRANT",
                    "L1_1_NOC_RING1_INCOMING_1_GRANT",
                )
            if has_counter("L1_0_UNPACKER_0") and has_counter("L1_0_UNPACKER_0_GRANT"):
                req = get_counter_series("L1_0_UNPACKER_0")
                grant = get_counter_series("L1_0_UNPACKER_0_GRANT")
                ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                l1_unpacker_bp = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }
            if has_counter("L1_0_UNIFIED_PACKER") and has_counter("L1_0_PORT1_GRANT"):
                req = get_counter_series("L1_0_UNIFIED_PACKER")
                grant = get_counter_series("L1_0_PORT1_GRANT")
                ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                l1_packer_port_bp = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }
            elif has_counter("L1_0_UNPACKER_1_ECC_PACK1") and has_counter("L1_0_PORT1_GRANT"):
                req = get_counter_series("L1_0_UNPACKER_1_ECC_PACK1")
                grant = get_counter_series("L1_0_PORT1_GRANT")
                ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                l1_packer_port_bp = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }

            # === NEW: Per-type instruction issue efficiency ===
            # Formula: ISSUED / AVAILABLE * 100 (per instruction type, on its primary thread)
            cfg_issue_eff = {}
            sync_issue_eff = {}
            thcon_issue_eff = {}
            move_issue_eff = {}
            math_instrn_issue_eff = {}
            unpack_instrn_issue_eff = {}
            pack_instrn_issue_eff = {}
            # Instruction availability rate = cycles available / ref_cnt * 100
            # Shows what % of time each instruction type was ready to issue
            if has_counter("CFG_INSTRN_AVAILABLE_0"):
                cfg_issue_eff = compute_util_metric("CFG_INSTRN_AVAILABLE_0")
            if has_counter("SYNC_INSTRN_AVAILABLE_0"):
                sync_issue_eff = compute_util_metric("SYNC_INSTRN_AVAILABLE_0")
            if has_counter("THCON_INSTRN_AVAILABLE_0"):
                thcon_issue_eff = compute_util_metric("THCON_INSTRN_AVAILABLE_0")
            if has_counter("MOVE_INSTRN_AVAILABLE_0"):
                move_issue_eff = compute_util_metric("MOVE_INSTRN_AVAILABLE_0")
            if has_counter("FPU_INSTRN_AVAILABLE_1"):
                math_instrn_issue_eff = compute_util_metric("FPU_INSTRN_AVAILABLE_1")
            if has_counter("UNPACK_INSTRN_AVAILABLE_0"):
                unpack_instrn_issue_eff = compute_util_metric("UNPACK_INSTRN_AVAILABLE_0")
            if has_counter("PACK_INSTRN_AVAILABLE_2"):
                pack_instrn_issue_eff = compute_util_metric("PACK_INSTRN_AVAILABLE_2")

            # === NEW: Stall breakdown (% of total stalls per thread) ===
            thcon_stall_pct = {}
            move_stall_pct = {}
            mmio_stall_pct = {}
            sfpu_stall_pct = {}
            if has_counter("WAITING_FOR_THCON_IDLE_0") and has_counter("THREAD_STALLS_0"):
                thcon_stall_pct = compute_ratio_metric("WAITING_FOR_THCON_IDLE_0", "THREAD_STALLS_0")
            if has_counter("WAITING_FOR_MOVE_IDLE_0") and has_counter("THREAD_STALLS_0"):
                move_stall_pct = compute_ratio_metric("WAITING_FOR_MOVE_IDLE_0", "THREAD_STALLS_0")
            if has_counter("WAITING_FOR_MMIO_IDLE_1") and has_counter("THREAD_STALLS_1"):
                mmio_stall_pct = compute_ratio_metric("WAITING_FOR_MMIO_IDLE_1", "THREAD_STALLS_1")
            if has_counter("WAITING_FOR_SFPU_IDLE_1") and has_counter("THREAD_STALLS_1"):
                sfpu_stall_pct = compute_ratio_metric("WAITING_FOR_SFPU_IDLE_1", "THREAD_STALLS_1")

            # === NEW: Write port blocking ===
            srcb_blocked = {}
            srca_write_eff = {}
            srcb_write_eff = {}
            if has_counter("SRCB_WRITE_AVAILABLE") and has_counter("SRCB_WRITE_NOT_BLOCKED_PORT"):
                srcb_blocked = compute_complement_metric("SRCB_WRITE_NOT_BLOCKED_PORT", "SRCB_WRITE_AVAILABLE")
            if has_counter("SRCA_WRITE_ACTUAL") and has_counter("SRCA_WRITE_AVAILABLE"):
                srca_write_eff = compute_ratio_metric("SRCA_WRITE_ACTUAL", "SRCA_WRITE_AVAILABLE")
            if has_counter("SRCB_WRITE_ACTUAL") and has_counter("SRCB_WRITE_AVAILABLE"):
                srcb_write_eff = compute_ratio_metric("SRCB_WRITE_ACTUAL", "SRCB_WRITE_AVAILABLE")

            # === NEW: Fidelity analysis ===
            hifi4_rate = {}
            fidelity_overhead = {}
            if has_counter("MATH_INSTRN_STARTED") and has_counter("INSTRN_2_HF_CYCLES") and has_counter("INSTRN_1_HF_CYCLE"):
                total = get_counter_series("MATH_INSTRN_STARTED")
                hf2 = get_counter_series("INSTRN_2_HF_CYCLES")
                hf1 = get_counter_series("INSTRN_1_HF_CYCLE")
                hf4 = total - hf2 - hf1
                ratio = (hf4 / total * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                hifi4_rate = {
                    "min": grouped.min().to_dict(),
                    "median": grouped.median().to_dict(),
                    "max": grouped.max().to_dict(),
                    "avg": grouped.mean().to_dict(),
                }
            if has_counter("FIDELITY_PHASE_STALLS"):
                fidelity_overhead = compute_util_metric("FIDELITY_PHASE_STALLS")

            # === NEW: Packer engine granularity ===
            packer_engine_0_util = {}
            packer_engine_1_util = {}
            packer_engine_2_util = {}
            if has_counter("PACKER_BUSY_0"):
                packer_engine_0_util = compute_util_metric("PACKER_BUSY_0")
            if has_counter("PACKER_BUSY_1"):
                packer_engine_1_util = compute_util_metric("PACKER_BUSY_1")
            if has_counter("PACKER_BUSY_2"):
                packer_engine_2_util = compute_util_metric("PACKER_BUSY_2")

            # === NEW: Low priority waits ===
            mmio_wait = {}
            sfpu_wait = {}
            thcon_wait = {}
            move_wait = {}
            risc_core_util = {}
            if has_counter("WAITING_FOR_MMIO_IDLE_0"):
                mmio_wait = compute_util_metric("WAITING_FOR_MMIO_IDLE_0")
            if has_counter("WAITING_FOR_SFPU_IDLE_1"):
                sfpu_wait = compute_util_metric("WAITING_FOR_SFPU_IDLE_1")
            if has_counter("WAITING_FOR_THCON_IDLE_0"):
                thcon_wait = compute_util_metric("WAITING_FOR_THCON_IDLE_0")
            if has_counter("WAITING_FOR_MOVE_IDLE_0"):
                move_wait = compute_util_metric("WAITING_FOR_MOVE_IDLE_0")
            if has_counter("L1_1_RISC_CORE"):
                risc_core_util = compute_util_metric("L1_1_RISC_CORE")

            # === L1 composite metrics (multi-counter) ===
            l1_total_bw = {}
            l1_rw_ratio = {}
            noc_asymmetry = {}
            l1_contention = {}
            unpacker_l1_eff = {}
            packer_l1_eff = {}
            noc_vs_compute = {}
            tdma_vs_noc = {}

            if has_counter("L1_0_UNPACKER_0") and has_counter("L1_0_NOC_RING0_OUTGOING_0"):
                # L1 Total Bandwidth Util: sum of all 8 port reqs / (8 * ref_cnt)
                packer_key = "L1_0_UNIFIED_PACKER" if has_counter("L1_0_UNIFIED_PACKER") else "L1_0_UNPACKER_1_ECC_PACK1"
                port_keys = ["L1_0_UNPACKER_0", packer_key, "L1_0_TDMA_BUNDLE_0_RISC", "L1_0_TDMA_BUNDLE_1_TRISC",
                             "L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1",
                             "L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1"]
                total_req = sum(get_counter_series(k) for k in port_keys if has_counter(k))
                ref = get_counter_ref_cnt("L1_0_UNPACKER_0")
                ratio = (total_req / (8 * ref) * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                l1_total_bw = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                               "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

                # L1 Read vs Write Ratio
                reads = get_counter_series("L1_0_UNPACKER_0") + get_counter_series("L1_0_NOC_RING0_OUTGOING_0") + get_counter_series("L1_0_NOC_RING0_OUTGOING_1")
                writes = get_counter_series(packer_key) + get_counter_series("L1_0_NOC_RING0_INCOMING_0") + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
                ratio = (reads / (reads + writes) * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                l1_rw_ratio = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                               "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

                # NOC Ring 0 Asymmetry
                noc_out = get_counter_series("L1_0_NOC_RING0_OUTGOING_0") + get_counter_series("L1_0_NOC_RING0_OUTGOING_1")
                noc_in = get_counter_series("L1_0_NOC_RING0_INCOMING_0") + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
                ratio = (noc_out / (noc_out + noc_in) * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                noc_asymmetry = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                                 "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

                # TDMA vs NOC L1 Share
                tdma = get_counter_series("L1_0_TDMA_BUNDLE_0_RISC") + get_counter_series("L1_0_TDMA_BUNDLE_1_TRISC")
                ratio = (tdma / (tdma + noc_out + noc_in) * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                tdma_vs_noc = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                               "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

            if has_counter("L1_0_UNPACKER_0_GRANT") and has_counter("L1_0_NOC_RING0_OUTGOING_0_GRANT"):
                # L1 Contention Index
                bp_pairs = [("L1_0_UNPACKER_0", "L1_0_UNPACKER_0_GRANT"),
                            ("L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_0_GRANT"),
                            ("L1_0_NOC_RING0_OUTGOING_1", "L1_0_NOC_RING0_OUTGOING_1_GRANT"),
                            ("L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_0_GRANT"),
                            ("L1_0_NOC_RING0_INCOMING_1", "L1_0_NOC_RING0_INCOMING_1_GRANT")]
                bp_sum = None
                bp_count = 0
                for req_k, grant_k in bp_pairs:
                    if has_counter(req_k) and has_counter(grant_k):
                        req_s = get_counter_series(req_k)
                        grant_s = get_counter_series(grant_k)
                        bp = ((req_s - grant_s) / req_s * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
                        bp_sum = bp if bp_sum is None else bp_sum + bp
                        bp_count += 1
                if bp_count > 0:
                    avg_bp = bp_sum / bp_count
                    grouped = avg_bp.groupby(level=["run_host_id", "trace_id_count"])
                    l1_contention = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                                     "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

            if has_counter("L1_0_UNPACKER_0_GRANT") and has_counter("UNPACK0_BUSY_THREAD0"):
                unpacker_l1_eff = compute_ratio_metric("L1_0_UNPACKER_0_GRANT", "UNPACK0_BUSY_THREAD0")

            if has_counter("L1_0_PORT1_GRANT") and has_counter("PACKER_BUSY"):
                # Packer port is shared (other traffic uses it too), so cap at 100%
                grant = get_counter_series("L1_0_PORT1_GRANT")
                busy = get_counter_series("PACKER_BUSY")
                ratio = (grant / busy * 100).clip(upper=100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                packer_l1_eff = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                                 "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

            if has_counter("FPU_COUNTER") and has_counter("L1_0_NOC_RING0_OUTGOING_0"):
                noc_total = get_counter_series("L1_0_NOC_RING0_OUTGOING_0") + get_counter_series("L1_0_NOC_RING0_OUTGOING_1") + \
                            get_counter_series("L1_0_NOC_RING0_INCOMING_0") + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
                fpu = get_counter_series("FPU_COUNTER")
                ratio = (noc_total / (fpu + noc_total) * 100).replace([float("inf"), -float("inf")], nan)
                grouped = ratio.groupby(level=["run_host_id", "trace_id_count"])
                noc_vs_compute = {"min": grouped.min().to_dict(), "median": grouped.median().to_dict(),
                                  "max": grouped.max().to_dict(), "avg": grouped.mean().to_dict()}

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

                # L1 Port 1 (arch-specific: BH unified packer, WH unpacker#1/ECC/pack1)
                assign_metric("L1 Packer Port Util", l1_packer_port_util)

                # L1 back-pressure (derived stall metrics: (req - grant) / req * 100)
                assign_metric("NOC Ring 0 Outgoing Backpressure", noc_r0_out_bp)
                assign_metric("NOC Ring 0 Incoming Backpressure", noc_r0_in_bp)
                assign_metric("NOC Ring 1 Outgoing Backpressure", noc_r1_out_bp)
                assign_metric("NOC Ring 1 Incoming Backpressure", noc_r1_in_bp)
                assign_metric("L1 Unpacker Backpressure", l1_unpacker_bp)
                assign_metric("L1 Packer Port Backpressure", l1_packer_port_bp)

                # Fidelity cycle breakdown
                assign_metric("HiFi2 Instrn Rate", hifi2_rate)
                assign_metric("LoFi Instrn Rate", lofi_rate)

                # Math pipeline stall breakdown
                assign_metric("Math Src Data Ready Rate", math_src_ready)
                assign_metric("SrcA Write Port Blocked Rate", srca_blocked)
                assign_metric("Dest Read Backpressure", dest_bp)
                assign_metric("Math Dest Write Port Stall Rate", math_dest_wr_stall)
                assign_metric("Math Scoreboard Stall Rate", math_scoreboard_stall)

                # Instruction issue rates
                assign_metric("Unpack Instrn Issue Rate T0", unpack_issue_rate, suffix="")
                assign_metric("Math Instrn Issue Rate T1", math_issue_rate, suffix="")
                assign_metric("Pack Instrn Issue Rate T2", pack_issue_rate, suffix="")

                # === NEW METRICS ===
                # Per-type instruction issue efficiency
                assign_metric("CFG Instrn Avail Rate T0", cfg_issue_eff)
                assign_metric("SYNC Instrn Avail Rate T0", sync_issue_eff)
                assign_metric("THCON Instrn Avail Rate T0", thcon_issue_eff)
                assign_metric("MOVE Instrn Avail Rate T0", move_issue_eff)
                assign_metric("MATH Instrn Avail Rate T1", math_instrn_issue_eff)
                assign_metric("UNPACK Instrn Avail Rate T0", unpack_instrn_issue_eff)
                assign_metric("PACK Instrn Avail Rate T2", pack_instrn_issue_eff)

                # Stall breakdown
                assign_metric("THCON Idle Stall Pct T0", thcon_stall_pct)
                assign_metric("MOVE Idle Stall Pct T0", move_stall_pct)
                assign_metric("MMIO Idle Stall Pct T1", mmio_stall_pct)
                assign_metric("SFPU Idle Stall Pct T1", sfpu_stall_pct)

                # Write port blocking
                assign_metric("SrcB Write Port Blocked Rate", srcb_blocked)
                assign_metric("SrcA Write Actual Efficiency", srca_write_eff)
                assign_metric("SrcB Write Actual Efficiency", srcb_write_eff)

                # Fidelity analysis
                assign_metric("HiFi4 Instrn Rate", hifi4_rate)
                assign_metric("Fidelity Phase Overhead", fidelity_overhead)

                # Packer engine granularity
                assign_metric("Packer Engine 0 Util", packer_engine_0_util)
                assign_metric("Packer Engine 1 Util", packer_engine_1_util)
                assign_metric("Packer Engine 2 Util", packer_engine_2_util)

                # Low priority waits
                assign_metric("MMIO Idle Wait T0", mmio_wait)
                assign_metric("SFPU Idle Wait T1", sfpu_wait)
                assign_metric("THCON Idle Wait T0", thcon_wait)
                assign_metric("MOVE Idle Wait T0", move_wait)
                assign_metric("RISC Core L1 Util", risc_core_util)

                # L1 composite metrics
                assign_metric("L1 Total Bandwidth Util", l1_total_bw)
                assign_metric("L1 Read vs Write Ratio", l1_rw_ratio)
                assign_metric("NOC Ring 0 Asymmetry", noc_asymmetry)
                assign_metric("L1 Contention Index", l1_contention)
                assign_metric("Unpacker L1 Efficiency", unpacker_l1_eff)
                assign_metric("Packer L1 Efficiency", packer_l1_eff)
                assign_metric("NOC vs Compute Balance", noc_vs_compute)
                assign_metric("TDMA vs NOC L1 Share", tdma_vs_noc)

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
                    if "value_MATH_INSTRN_STARTED" in eff_pivot.columns:
                        eff_pivot["Math Pipeline Utilization"] = eff_pivot.apply(
                            lambda x: safe_div(
                                x.get("value_MATH_INSTRN_STARTED", 0), x.get("value_MATH_INSTRN_AVAILABLE", 0)
                            ),
                            axis=1,
                        )
                    eff_pivot["Math-to-Pack Handoff Efficiency"] = eff_pivot.apply(
                        lambda x: min(100.0, safe_div(x.get("value_AVAILABLE_MATH", 0), x.get("value_PACKER_BUSY", 0))), axis=1
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
                    # L1 Port 1 (arch-specific: BH unified packer, WH unpacker#1/ECC/pack1)
                    eff_pivot["L1 Packer Port Util"] = eff_pivot.apply(
                        lambda x: safe_div(
                            x.get("value_L1_0_UNIFIED_PACKER", x.get("value_L1_0_UNPACKER_1_ECC_PACK1", 0)),
                            x.get("ref_cnt_L1_0_UNIFIED_PACKER", x.get("ref_cnt_L1_0_UNPACKER_1_ECC_PACK1", 0)),
                        ),
                        axis=1,
                    )

                    # L1 back-pressure: (req - grant) / req * 100
                    def safe_backpressure(req0_key, req1_key, grant0_key, grant1_key):
                        def fn(x):
                            r0 = x.get(req0_key, 0)
                            r1 = x.get(req1_key, 0)
                            g0 = x.get(grant0_key, 0)
                            g1 = x.get(grant1_key, 0)
                            total_req = r0 + r1
                            return max(0.0, (total_req - g0 - g1) / total_req * 100) if total_req > 0 else nan

                        return fn

                    eff_pivot["NOC Ring 0 Outgoing Backpressure"] = eff_pivot.apply(
                        safe_backpressure(
                            "value_L1_0_NOC_RING0_OUTGOING_0",
                            "value_L1_0_NOC_RING0_OUTGOING_1",
                            "value_L1_0_NOC_RING0_OUTGOING_0_GRANT",
                            "value_L1_0_NOC_RING0_OUTGOING_1_GRANT",
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 0 Incoming Backpressure"] = eff_pivot.apply(
                        safe_backpressure(
                            "value_L1_0_NOC_RING0_INCOMING_0",
                            "value_L1_0_NOC_RING0_INCOMING_1",
                            "value_L1_0_NOC_RING0_INCOMING_0_GRANT",
                            "value_L1_0_NOC_RING0_INCOMING_1_GRANT",
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 1 Outgoing Backpressure"] = eff_pivot.apply(
                        safe_backpressure(
                            "value_L1_1_NOC_RING1_OUTGOING_0",
                            "value_L1_1_NOC_RING1_OUTGOING_1",
                            "value_L1_1_NOC_RING1_OUTGOING_0_GRANT",
                            "value_L1_1_NOC_RING1_OUTGOING_1_GRANT",
                        ),
                        axis=1,
                    )
                    eff_pivot["NOC Ring 1 Incoming Backpressure"] = eff_pivot.apply(
                        safe_backpressure(
                            "value_L1_1_NOC_RING1_INCOMING_0",
                            "value_L1_1_NOC_RING1_INCOMING_1",
                            "value_L1_1_NOC_RING1_INCOMING_0_GRANT",
                            "value_L1_1_NOC_RING1_INCOMING_1_GRANT",
                        ),
                        axis=1,
                    )

                    def safe_single_bp(req_key, grant_key):
                        def fn(x):
                            r = x.get(req_key, 0)
                            g = x.get(grant_key, 0)
                            return max(0.0, (r - g) / r * 100) if r > 0 else nan

                        return fn

                    eff_pivot["L1 Unpacker Backpressure"] = eff_pivot.apply(
                        safe_single_bp("value_L1_0_UNPACKER_0", "value_L1_0_UNPACKER_0_GRANT"),
                        axis=1,
                    )
                    # L1 Packer Port: BH uses L1_0_UNIFIED_PACKER, WH uses L1_0_UNPACKER_1_ECC_PACK1
                    packer_req_key = (
                        "value_L1_0_UNIFIED_PACKER"
                        if "value_L1_0_UNIFIED_PACKER" in eff_pivot.columns
                        else "value_L1_0_UNPACKER_1_ECC_PACK1"
                    )
                    eff_pivot["L1 Packer Port Backpressure"] = eff_pivot.apply(
                        safe_single_bp(packer_req_key, "value_L1_0_PORT1_GRANT"),
                        axis=1,
                    )

                    # === NEW: Per-type instruction issue efficiency ===
                    def safe_ratio(num_key, den_key):
                        def fn(x):
                            n = x.get(num_key, 0)
                            d = x.get(den_key, 0)
                            return (n / d * 100) if d > 0 else nan
                        return fn

                    def safe_complement(counter_key, total_key):
                        def fn(x):
                            v = x.get(counter_key, 0)
                            t = x.get(total_key, 0)
                            return max(0.0, (t - v) / t * 100) if t > 0 else nan
                        return fn

                    def safe_util(counter_key, ref_key):
                        def fn(x):
                            v = x.get(counter_key, 0)
                            r = x.get(ref_key, 0)
                            return (v / r * 100) if r > 0 else nan
                        return fn

                    # Instruction availability rate = cycles available / ref_cnt * 100
                    # Shows what % of time each instruction type was ready to issue
                    eff_pivot["CFG Instrn Avail Rate T0"] = eff_pivot.apply(
                        safe_util("value_CFG_INSTRN_AVAILABLE_0", "ref_cnt_CFG_INSTRN_AVAILABLE_0"), axis=1)
                    eff_pivot["SYNC Instrn Avail Rate T0"] = eff_pivot.apply(
                        safe_util("value_SYNC_INSTRN_AVAILABLE_0", "ref_cnt_SYNC_INSTRN_AVAILABLE_0"), axis=1)
                    eff_pivot["THCON Instrn Avail Rate T0"] = eff_pivot.apply(
                        safe_util("value_THCON_INSTRN_AVAILABLE_0", "ref_cnt_THCON_INSTRN_AVAILABLE_0"), axis=1)
                    eff_pivot["MOVE Instrn Avail Rate T0"] = eff_pivot.apply(
                        safe_util("value_MOVE_INSTRN_AVAILABLE_0", "ref_cnt_MOVE_INSTRN_AVAILABLE_0"), axis=1)
                    eff_pivot["MATH Instrn Avail Rate T1"] = eff_pivot.apply(
                        safe_util("value_FPU_INSTRN_AVAILABLE_1", "ref_cnt_FPU_INSTRN_AVAILABLE_1"), axis=1)
                    eff_pivot["UNPACK Instrn Avail Rate T0"] = eff_pivot.apply(
                        safe_util("value_UNPACK_INSTRN_AVAILABLE_0", "ref_cnt_UNPACK_INSTRN_AVAILABLE_0"), axis=1)
                    eff_pivot["PACK Instrn Avail Rate T2"] = eff_pivot.apply(
                        safe_util("value_PACK_INSTRN_AVAILABLE_2", "ref_cnt_PACK_INSTRN_AVAILABLE_2"), axis=1)

                    # Stall breakdown (% of total stalls per thread)
                    eff_pivot["THCON Idle Stall Pct T0"] = eff_pivot.apply(
                        safe_ratio("value_WAITING_FOR_THCON_IDLE_0", "value_THREAD_STALLS_0"), axis=1)
                    eff_pivot["MOVE Idle Stall Pct T0"] = eff_pivot.apply(
                        safe_ratio("value_WAITING_FOR_MOVE_IDLE_0", "value_THREAD_STALLS_0"), axis=1)
                    eff_pivot["MMIO Idle Stall Pct T1"] = eff_pivot.apply(
                        safe_ratio("value_WAITING_FOR_MMIO_IDLE_1", "value_THREAD_STALLS_1"), axis=1)
                    eff_pivot["SFPU Idle Stall Pct T1"] = eff_pivot.apply(
                        safe_ratio("value_WAITING_FOR_SFPU_IDLE_1", "value_THREAD_STALLS_1"), axis=1)

                    # Write port blocking
                    eff_pivot["SrcA Write Port Blocked Rate"] = eff_pivot.apply(
                        safe_complement("value_SRCA_WRITE_NOT_BLOCKED_OVR", "value_SRCA_WRITE_AVAILABLE"), axis=1)
                    eff_pivot["SrcB Write Port Blocked Rate"] = eff_pivot.apply(
                        safe_complement("value_SRCB_WRITE_NOT_BLOCKED_PORT", "value_SRCB_WRITE_AVAILABLE"), axis=1)
                    eff_pivot["SrcA Write Actual Efficiency"] = eff_pivot.apply(
                        safe_ratio("value_SRCA_WRITE_ACTUAL", "value_SRCA_WRITE_AVAILABLE"), axis=1)
                    if "value_SRCB_WRITE_ACTUAL" in eff_pivot.columns:
                        eff_pivot["SrcB Write Actual Efficiency"] = eff_pivot.apply(
                            safe_ratio("value_SRCB_WRITE_ACTUAL", "value_SRCB_WRITE_AVAILABLE"), axis=1)

                    # Dest read and math stall metrics
                    def safe_bp_single(req_key, grant_key):
                        def fn(x):
                            r = x.get(req_key, 0)
                            g = x.get(grant_key, 0)
                            return max(0.0, (r - g) / r * 100) if r > 0 else nan
                        return fn
                    eff_pivot["Dest Read Backpressure"] = eff_pivot.apply(
                        safe_bp_single("value_PACKER_DEST_READ_AVAILABLE", "value_DEST_READ_GRANTED_0"), axis=1)
                    eff_pivot["Math Dest Write Port Stall Rate"] = eff_pivot.apply(
                        safe_complement("value_MATH_NOT_STALLED_DEST_WR_PORT", "value_MATH_INSTRN_AVAILABLE"), axis=1)
                    eff_pivot["Math Scoreboard Stall Rate"] = eff_pivot.apply(
                        safe_complement("value_AVAILABLE_MATH", "value_MATH_INSTRN_AVAILABLE"), axis=1)

                    # Instruction issue rates (per cycle, not %)
                    eff_pivot["Unpack Instrn Issue Rate T0"] = eff_pivot.apply(
                        lambda x: x.get("value_UNPACK_INSTRN_ISSUED_0", 0) / x.get("ref_cnt_UNPACK_INSTRN_ISSUED_0", 1)
                        if x.get("ref_cnt_UNPACK_INSTRN_ISSUED_0", 0) > 0 else nan, axis=1)
                    eff_pivot["Math Instrn Issue Rate T1"] = eff_pivot.apply(
                        lambda x: x.get("value_FPU_INSTRN_ISSUED_1", 0) / x.get("ref_cnt_FPU_INSTRN_ISSUED_1", 1)
                        if x.get("ref_cnt_FPU_INSTRN_ISSUED_1", 0) > 0 else nan, axis=1)
                    eff_pivot["Pack Instrn Issue Rate T2"] = eff_pivot.apply(
                        lambda x: x.get("value_PACK_INSTRN_ISSUED_2", 0) / x.get("ref_cnt_PACK_INSTRN_ISSUED_2", 1)
                        if x.get("ref_cnt_PACK_INSTRN_ISSUED_2", 0) > 0 else nan, axis=1)

                    # Fidelity analysis
                    def hifi4_rate_fn(x):
                        total = x.get("value_MATH_INSTRN_STARTED", 0)
                        hf2 = x.get("value_INSTRN_2_HF_CYCLES", 0)
                        hf1 = x.get("value_INSTRN_1_HF_CYCLE", 0)
                        return max(0.0, (total - hf2 - hf1) / total * 100) if total > 0 else nan
                    eff_pivot["HiFi4 Instrn Rate"] = eff_pivot.apply(hifi4_rate_fn, axis=1)
                    eff_pivot["Fidelity Phase Overhead"] = eff_pivot.apply(
                        safe_util("value_FIDELITY_PHASE_STALLS", "ref_cnt_FIDELITY_PHASE_STALLS"), axis=1)

                    # Packer engine granularity (WH only — BH has PACK_COUNT=1, counters not collected)
                    if "value_PACKER_BUSY_0" in eff_pivot.columns:
                        eff_pivot["Packer Engine 0 Util"] = eff_pivot.apply(
                            safe_util("value_PACKER_BUSY_0", "ref_cnt_PACKER_BUSY_0"), axis=1)
                        eff_pivot["Packer Engine 1 Util"] = eff_pivot.apply(
                            safe_util("value_PACKER_BUSY_1", "ref_cnt_PACKER_BUSY_1"), axis=1)
                        eff_pivot["Packer Engine 2 Util"] = eff_pivot.apply(
                            safe_util("value_PACKER_BUSY_2", "ref_cnt_PACKER_BUSY_2"), axis=1)

                    # Low priority waits
                    eff_pivot["MMIO Idle Wait T0"] = eff_pivot.apply(
                        safe_util("value_WAITING_FOR_MMIO_IDLE_0", "ref_cnt_WAITING_FOR_MMIO_IDLE_0"), axis=1)
                    eff_pivot["SFPU Idle Wait T1"] = eff_pivot.apply(
                        safe_util("value_WAITING_FOR_SFPU_IDLE_1", "ref_cnt_WAITING_FOR_SFPU_IDLE_1"), axis=1)
                    eff_pivot["THCON Idle Wait T0"] = eff_pivot.apply(
                        safe_util("value_WAITING_FOR_THCON_IDLE_0", "ref_cnt_WAITING_FOR_THCON_IDLE_0"), axis=1)
                    eff_pivot["MOVE Idle Wait T0"] = eff_pivot.apply(
                        safe_util("value_WAITING_FOR_MOVE_IDLE_0", "ref_cnt_WAITING_FOR_MOVE_IDLE_0"), axis=1)
                    eff_pivot["RISC Core L1 Util"] = eff_pivot.apply(
                        safe_util("value_L1_1_RISC_CORE", "ref_cnt_L1_1_RISC_CORE"), axis=1)

                    # === L1 composite metrics (multi-counter) ===
                    def l1_total_bw(x):
                        """Sum of all 8 L1_0 port req counts / (8 * ref_cnt)."""
                        ports = ["value_L1_0_UNPACKER_0",
                                 "value_L1_0_UNIFIED_PACKER" if "value_L1_0_UNIFIED_PACKER" in eff_pivot.columns
                                 else "value_L1_0_UNPACKER_1_ECC_PACK1",
                                 "value_L1_0_TDMA_BUNDLE_0_RISC", "value_L1_0_TDMA_BUNDLE_1_TRISC",
                                 "value_L1_0_NOC_RING0_OUTGOING_0", "value_L1_0_NOC_RING0_OUTGOING_1",
                                 "value_L1_0_NOC_RING0_INCOMING_0", "value_L1_0_NOC_RING0_INCOMING_1"]
                        total = sum(x.get(p, 0) for p in ports)
                        ref = x.get("ref_cnt_L1_0_UNPACKER_0", 0)
                        return (total / (8 * ref) * 100) if ref > 0 else nan
                    eff_pivot["L1 Total Bandwidth Util"] = eff_pivot.apply(l1_total_bw, axis=1)

                    def l1_rw_ratio(x):
                        """(read ports) / (write ports). Read = Unpacker + NOC Out, Write = Packer + NOC In."""
                        reads = (x.get("value_L1_0_UNPACKER_0", 0) +
                                 x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) +
                                 x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0))
                        writes = (x.get("value_L1_0_UNIFIED_PACKER",
                                  x.get("value_L1_0_UNPACKER_1_ECC_PACK1", 0)) +
                                  x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) +
                                  x.get("value_L1_0_NOC_RING0_INCOMING_1", 0))
                        total = reads + writes
                        return (reads / total * 100) if total > 0 else nan
                    eff_pivot["L1 Read vs Write Ratio"] = eff_pivot.apply(l1_rw_ratio, axis=1)

                    def noc_asymmetry(x):
                        """NOC outgoing / (outgoing + incoming). 50% = balanced."""
                        out0 = x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0)
                        in0 = x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0)
                        total = out0 + in0
                        return (out0 / total * 100) if total > 0 else nan
                    eff_pivot["NOC Ring 0 Asymmetry"] = eff_pivot.apply(noc_asymmetry, axis=1)

                    def l1_contention_index(x):
                        """Average backpressure across all active L1_0 ports."""
                        ports = [
                            ("value_L1_0_UNPACKER_0", "value_L1_0_UNPACKER_0_GRANT"),
                            ("value_L1_0_NOC_RING0_OUTGOING_0", "value_L1_0_NOC_RING0_OUTGOING_0_GRANT"),
                            ("value_L1_0_NOC_RING0_OUTGOING_1", "value_L1_0_NOC_RING0_OUTGOING_1_GRANT"),
                            ("value_L1_0_NOC_RING0_INCOMING_0", "value_L1_0_NOC_RING0_INCOMING_0_GRANT"),
                            ("value_L1_0_NOC_RING0_INCOMING_1", "value_L1_0_NOC_RING0_INCOMING_1_GRANT"),
                        ]
                        bp_values = []
                        for req_key, grant_key in ports:
                            req = x.get(req_key, 0)
                            grant = x.get(grant_key, 0)
                            if req > 0:
                                bp_values.append(max(0.0, (req - grant) / req * 100))
                        return sum(bp_values) / len(bp_values) if bp_values else nan
                    eff_pivot["L1 Contention Index"] = eff_pivot.apply(l1_contention_index, axis=1)

                    def unpacker_l1_eff(x):
                        """L1 grant to unpacker / unpacker busy cycles. How well L1 serves the unpacker."""
                        grant = x.get("value_L1_0_UNPACKER_0_GRANT", 0)
                        busy = x.get("value_UNPACK0_BUSY_THREAD0", 0)
                        return (grant / busy * 100) if busy > 0 else nan
                    eff_pivot["Unpacker L1 Efficiency"] = eff_pivot.apply(unpacker_l1_eff, axis=1)

                    def packer_l1_eff(x):
                        """L1 grant to packer port / packer busy cycles. Capped at 100%."""
                        grant = x.get("value_L1_0_PORT1_GRANT", 0)
                        busy = x.get("value_PACKER_BUSY", 0)
                        return min(100.0, grant / busy * 100) if busy > 0 else nan
                    eff_pivot["Packer L1 Efficiency"] = eff_pivot.apply(packer_l1_eff, axis=1)

                    def noc_vs_compute(x):
                        """NOC cycles / (FPU + NOC cycles). >50% = NOC-bound, <50% = compute-bound."""
                        noc = (x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) +
                               x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0) +
                               x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) +
                               x.get("value_L1_0_NOC_RING0_INCOMING_1", 0))
                        fpu = x.get("value_FPU_COUNTER", 0)
                        total = fpu + noc
                        return (noc / total * 100) if total > 0 else nan
                    eff_pivot["NOC vs Compute Balance"] = eff_pivot.apply(noc_vs_compute, axis=1)

                    def tdma_vs_noc_share(x):
                        """TDMA L1 share = TDMA / (TDMA + NOC). Shows RISC vs NOC memory traffic split."""
                        tdma = (x.get("value_L1_0_TDMA_BUNDLE_0_RISC", 0) +
                                x.get("value_L1_0_TDMA_BUNDLE_1_TRISC", 0))
                        noc = (x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) +
                               x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0) +
                               x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) +
                               x.get("value_L1_0_NOC_RING0_INCOMING_1", 0))
                        total = tdma + noc
                        return (tdma / total * 100) if total > 0 else nan
                    eff_pivot["TDMA vs NOC L1 Share"] = eff_pivot.apply(tdma_vs_noc_share, axis=1)

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
                        "L1 Packer Port Util",
                        "NOC Ring 0 Outgoing Backpressure",
                        "NOC Ring 0 Incoming Backpressure",
                        "NOC Ring 1 Outgoing Backpressure",
                        "NOC Ring 1 Incoming Backpressure",
                        "L1 Unpacker Backpressure",
                        "L1 Packer Port Backpressure",
                        "HiFi2 Instrn Rate",
                        "LoFi Instrn Rate",
                        "Math Src Data Ready Rate",
                        "SrcA Write Port Blocked Rate",
                        "Dest Read Backpressure",
                        "Math Dest Write Port Stall Rate",
                        "Math Scoreboard Stall Rate",
                        # NEW metrics
                        "CFG Instrn Avail Rate T0",
                        "SYNC Instrn Avail Rate T0",
                        "THCON Instrn Avail Rate T0",
                        "MOVE Instrn Avail Rate T0",
                        "MATH Instrn Avail Rate T1",
                        "UNPACK Instrn Avail Rate T0",
                        "PACK Instrn Avail Rate T2",
                        "THCON Idle Stall Pct T0",
                        "MOVE Idle Stall Pct T0",
                        "MMIO Idle Stall Pct T1",
                        "SFPU Idle Stall Pct T1",
                        "SrcB Write Port Blocked Rate",
                        "SrcA Write Actual Efficiency",
                        "SrcB Write Actual Efficiency",
                        "HiFi4 Instrn Rate",
                        "Fidelity Phase Overhead",
                        "Packer Engine 0 Util",
                        "Packer Engine 1 Util",
                        "Packer Engine 2 Util",
                        "MMIO Idle Wait T0",
                        "SFPU Idle Wait T1",
                        "THCON Idle Wait T0",
                        "MOVE Idle Wait T0",
                        "RISC Core L1 Util",
                        # L1 composite metrics
                        "L1 Total Bandwidth Util",
                        "L1 Read vs Write Ratio",
                        "NOC Ring 0 Asymmetry",
                        "L1 Contention Index",
                        "Unpacker L1 Efficiency",
                        "Packer L1 Efficiency",
                        "NOC vs Compute Balance",
                        "TDMA vs NOC L1 Share",
                    ]
                    # Non-percentage metrics (raw rates)
                    _ipc_metric_names = [
                        "Thread 0 IPC",
                        "Thread 1 IPC",
                        "Thread 2 IPC",
                        "Unpack Instrn Issue Rate T0",
                        "Math Instrn Issue Rate T1",
                        "Pack Instrn Issue Rate T2",
                    ]

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
