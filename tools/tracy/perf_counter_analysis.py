#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from math import nan
from loguru import logger

OpDict = Dict[str, Any]
DeviceOpsDict = Dict[int, List[OpDict]]

# Counter type enum from perf_counters.hpp — auto-generated, must match C++ enum order
COUNTER_TYPE_NAMES = {
    0: "UNDEF",
    # FPU Group
    1: "FPU_COUNTER",
    2: "SFPU_COUNTER",
    3: "MATH_COUNTER",
    # TDMA_UNPACK Group (11 req + 11 grant = 22)
    4: "MATH_SRC_DATA_READY",
    5: "DATA_HAZARD_STALLS_MOVD2A",
    6: "FIDELITY_PHASE_STALLS",
    7: "MATH_INSTRN_STARTED",
    8: "MATH_INSTRN_AVAILABLE",
    9: "SRCB_WRITE_AVAILABLE",
    10: "SRCA_WRITE_AVAILABLE",
    11: "UNPACK0_BUSY_THREAD0",
    12: "UNPACK1_BUSY_THREAD0",
    13: "UNPACK0_BUSY_THREAD1",
    14: "UNPACK1_BUSY_THREAD1",
    15: "SRCB_WRITE",
    16: "SRCA_WRITE",
    # TDMA_PACK Group (8 req + 6 grant = 14)
    17: "PACKER_DEST_READ_AVAILABLE",
    18: "PACKER_BUSY",
    19: "AVAILABLE_MATH",
    # INSTRN_THREAD Group (61 req + 24 grant = 85)
    20: "CFG_INSTRN_AVAILABLE_0",
    21: "CFG_INSTRN_AVAILABLE_1",
    22: "CFG_INSTRN_AVAILABLE_2",
    23: "SYNC_INSTRN_AVAILABLE_0",
    24: "SYNC_INSTRN_AVAILABLE_1",
    25: "SYNC_INSTRN_AVAILABLE_2",
    26: "THCON_INSTRN_AVAILABLE_0",
    27: "THCON_INSTRN_AVAILABLE_1",
    28: "THCON_INSTRN_AVAILABLE_2",
    29: "XSEARCH_INSTRN_AVAILABLE_0",
    30: "XSEARCH_INSTRN_AVAILABLE_1",
    31: "XSEARCH_INSTRN_AVAILABLE_2",
    32: "MOVE_INSTRN_AVAILABLE_0",
    33: "MOVE_INSTRN_AVAILABLE_1",
    34: "MOVE_INSTRN_AVAILABLE_2",
    35: "FPU_INSTRN_AVAILABLE_0",
    36: "FPU_INSTRN_AVAILABLE_1",
    37: "FPU_INSTRN_AVAILABLE_2",
    38: "UNPACK_INSTRN_AVAILABLE_0",
    39: "UNPACK_INSTRN_AVAILABLE_1",
    40: "UNPACK_INSTRN_AVAILABLE_2",
    41: "PACK_INSTRN_AVAILABLE_0",
    42: "PACK_INSTRN_AVAILABLE_1",
    43: "PACK_INSTRN_AVAILABLE_2",
    44: "THREAD_STALLS_0",
    45: "THREAD_STALLS_1",
    46: "THREAD_STALLS_2",
    47: "WAITING_FOR_SRCA_CLEAR",
    48: "WAITING_FOR_SRCB_CLEAR",
    49: "WAITING_FOR_SRCA_VALID",
    50: "WAITING_FOR_SRCB_VALID",
    51: "WAITING_FOR_THCON_IDLE_0",
    52: "WAITING_FOR_THCON_IDLE_1",
    53: "WAITING_FOR_THCON_IDLE_2",
    54: "WAITING_FOR_UNPACK_IDLE_0",
    55: "WAITING_FOR_UNPACK_IDLE_1",
    56: "WAITING_FOR_UNPACK_IDLE_2",
    57: "WAITING_FOR_PACK_IDLE_0",
    58: "WAITING_FOR_PACK_IDLE_1",
    59: "WAITING_FOR_PACK_IDLE_2",
    60: "WAITING_FOR_MATH_IDLE_0",
    61: "WAITING_FOR_MATH_IDLE_1",
    62: "WAITING_FOR_MATH_IDLE_2",
    63: "WAITING_FOR_NONZERO_SEM_0",
    64: "WAITING_FOR_NONZERO_SEM_1",
    65: "WAITING_FOR_NONZERO_SEM_2",
    66: "WAITING_FOR_NONFULL_SEM_0",
    67: "WAITING_FOR_NONFULL_SEM_1",
    68: "WAITING_FOR_NONFULL_SEM_2",
    69: "WAITING_FOR_MOVE_IDLE_0",
    70: "WAITING_FOR_MOVE_IDLE_1",
    71: "WAITING_FOR_MOVE_IDLE_2",
    72: "WAITING_FOR_MMIO_IDLE_0",
    73: "WAITING_FOR_MMIO_IDLE_1",
    74: "WAITING_FOR_MMIO_IDLE_2",
    75: "WAITING_FOR_SFPU_IDLE_0",
    76: "WAITING_FOR_SFPU_IDLE_1",
    77: "WAITING_FOR_SFPU_IDLE_2",
    78: "THREAD_INSTRUCTIONS_0",
    79: "THREAD_INSTRUCTIONS_1",
    80: "THREAD_INSTRUCTIONS_2",
    # L1 Bank 0 req (ports 0-7)
    81: "L1_0_UNPACKER_0",
    82: "L1_0_UNPACKER_1_ECC_PACK1",
    83: "L1_0_TDMA_BUNDLE_0_RISC",
    84: "L1_0_TDMA_BUNDLE_1_TRISC",
    85: "L1_0_NOC_RING0_OUTGOING_0",
    86: "L1_0_NOC_RING0_OUTGOING_1",
    87: "L1_0_NOC_RING0_INCOMING_0",
    88: "L1_0_NOC_RING0_INCOMING_1",
    # L1 Bank 1 req (ports 8-15)
    89: "L1_1_TDMA_PACKER_2",
    90: "L1_1_EXT_UNPACKER_1",
    91: "L1_1_EXT_UNPACKER_2",
    92: "L1_1_EXT_UNPACKER_3",
    93: "L1_1_NOC_RING1_OUTGOING_0",
    94: "L1_1_NOC_RING1_OUTGOING_1",
    95: "L1_1_NOC_RING1_INCOMING_0",
    96: "L1_1_NOC_RING1_INCOMING_1",
    # Blackhole-specific L1 ports
    97: "L1_0_UNIFIED_PACKER",
    98: "L1_1_RISC_CORE",
    # L1 Bank 0 grant counters
    99: "L1_0_UNPACKER_0_GRANT",
    100: "L1_0_PORT1_GRANT",
    101: "L1_0_TDMA_BUNDLE_0_GRANT",
    102: "L1_0_TDMA_BUNDLE_1_GRANT",
    103: "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    104: "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    105: "L1_0_NOC_RING0_INCOMING_0_GRANT",
    106: "L1_0_NOC_RING0_INCOMING_1_GRANT",
    # L1 Bank 1 grant counters
    107: "L1_1_PORT8_GRANT",
    108: "L1_1_EXT_UNPACKER_1_GRANT",
    109: "L1_1_EXT_UNPACKER_2_GRANT",
    110: "L1_1_EXT_UNPACKER_3_GRANT",
    111: "L1_1_NOC_RING1_OUTGOING_0_GRANT",
    112: "L1_1_NOC_RING1_OUTGOING_1_GRANT",
    113: "L1_1_NOC_RING1_INCOMING_0_GRANT",
    114: "L1_1_NOC_RING1_INCOMING_1_GRANT",
    # INSTRN_THREAD grant counters (instruction issue counts)
    115: "CFG_INSTRN_ISSUED_0",
    116: "CFG_INSTRN_ISSUED_1",
    117: "CFG_INSTRN_ISSUED_2",
    118: "SYNC_INSTRN_ISSUED_0",
    119: "SYNC_INSTRN_ISSUED_1",
    120: "SYNC_INSTRN_ISSUED_2",
    121: "THCON_INSTRN_ISSUED_0",
    122: "THCON_INSTRN_ISSUED_1",
    123: "THCON_INSTRN_ISSUED_2",
    124: "XSEARCH_INSTRN_ISSUED_0",
    125: "XSEARCH_INSTRN_ISSUED_1",
    126: "XSEARCH_INSTRN_ISSUED_2",
    127: "MOVE_INSTRN_ISSUED_0",
    128: "MOVE_INSTRN_ISSUED_1",
    129: "MOVE_INSTRN_ISSUED_2",
    130: "FPU_INSTRN_ISSUED_0",
    131: "FPU_INSTRN_ISSUED_1",
    132: "FPU_INSTRN_ISSUED_2",
    133: "UNPACK_INSTRN_ISSUED_0",
    134: "UNPACK_INSTRN_ISSUED_1",
    135: "UNPACK_INSTRN_ISSUED_2",
    136: "PACK_INSTRN_ISSUED_0",
    137: "PACK_INSTRN_ISSUED_1",
    138: "PACK_INSTRN_ISSUED_2",
    # TDMA_UNPACK grant counters
    139: "INSTRN_2_HF_CYCLES",
    140: "INSTRN_1_HF_CYCLE",
    141: "SRCB_WRITE_ACTUAL",
    142: "SRCA_WRITE_NOT_BLOCKED_OVR",
    143: "SRCA_WRITE_ACTUAL",
    144: "SRCB_WRITE_NOT_BLOCKED_PORT",
    145: "SRCA_WRITE_THREAD0",
    146: "SRCB_WRITE_THREAD0",
    147: "SRCA_WRITE_THREAD1",
    148: "SRCB_WRITE_THREAD1",
    149: "MATH_INSTRN_NOT_BLOCKED_SRC",
    # TDMA_PACK additional req counters
    150: "PACKER_DEST_READ_1",
    151: "PACKER_DEST_READ_2",
    152: "PACKER_DEST_READ_3",
    153: "PACKER_BUSY_0",
    154: "PACKER_BUSY_1",
    155: "PACKER_BUSY_2",
    # TDMA_PACK grant counters
    156: "DEST_READ_GRANTED_0",
    157: "DEST_READ_GRANTED_1",
    158: "DEST_READ_GRANTED_2",
    159: "DEST_READ_GRANTED_3",
    160: "MATH_NOT_STALLED_DEST_WR_PORT",
    # L1 Bank 4 req counters (BH only, mux position 4, misc ports 32-39)
    161: "L1_4_MISC_PORT_0",
    162: "L1_4_MISC_PORT_1",
    163: "L1_4_MISC_PORT_2",
    164: "L1_4_MISC_PORT_3",
    165: "L1_4_MISC_PORT_4",
    166: "L1_4_MISC_PORT_5",
    167: "L1_4_MISC_PORT_6",
    168: "L1_4_MISC_PORT_7",
    # L1 Bank 4 grant counters
    169: "L1_4_MISC_PORT_0_GRANT",
    170: "L1_4_MISC_PORT_1_GRANT",
    171: "L1_4_MISC_PORT_2_GRANT",
    172: "L1_4_MISC_PORT_3_GRANT",
    173: "L1_4_MISC_PORT_4_GRANT",
    174: "L1_4_MISC_PORT_5_GRANT",
    175: "L1_4_MISC_PORT_6_GRANT",
    176: "L1_4_MISC_PORT_7_GRANT",
}


def extract_perf_counters(events: List[Any]) -> Optional[pd.DataFrame]:
    # If perf counter data exists, extract relevant columns and return as a dataframe
    EVENT_METADATA_IDX = 0
    EVENT_TIMESTAMP_IDX = 1
    EVENT_RISC_TYPE_IDX = 3
    EVENT_CORE_COORDS_IDX = 4
    PERF_COUNTER_ID = 9090

    try:
        # Process events: extract metadata, add timestamp and coords
        perf_counter_events = []
        for event in events:
            metadata = event[EVENT_METADATA_IDX]
            if metadata["id"] == PERF_COUNTER_ID:
                meta_dict = json.loads(metadata["meta_data"].replace(";", ",").replace("'", '"'))

                # Decode counter type to human-readable name
                counter_type_raw = meta_dict.get("counter type", 0)
                # Handle both integer ID and string name formats
                if isinstance(counter_type_raw, str):
                    counter_type_name = counter_type_raw
                else:
                    counter_type_name = COUNTER_TYPE_NAMES.get(counter_type_raw, f"UNKNOWN_{counter_type_raw}")

                perf_counter_events.append(
                    {
                        "run_host_id": metadata["run_host_id"],
                        "trace_id_count": metadata["trace_id_count"],
                        "record time": event[EVENT_TIMESTAMP_IDX],
                        "core_x": event[EVENT_CORE_COORDS_IDX][0],
                        "core_y": event[EVENT_CORE_COORDS_IDX][1],
                        "risc_type": event[EVENT_RISC_TYPE_IDX],
                        "counter type": counter_type_name,  # Use human-readable name
                        "value": meta_dict.get("value", 0),
                        "ref cnt": meta_dict.get("ref cnt", 0),
                    }
                )

        if perf_counter_events:
            return pd.DataFrame(perf_counter_events)
    except (KeyError, TypeError, AttributeError) as e:
        logger.exception("Failed to extract perf counter events: %s", e)
    return None


def print_counter_statistics_summary(perf_counter_df: pd.DataFrame, device_id: int) -> None:
    """Print statistics for all raw performance counters."""
    if perf_counter_df is None or perf_counter_df.empty:
        return

    print("\n" + "=" * 100)
    print(f"PERFORMANCE COUNTER STATISTICS - DEVICE {device_id}")
    print("=" * 100)

    # Group by operation
    grouped = perf_counter_df.groupby(["run_host_id", "trace_id_count"])
    total_ops = len(grouped)

    print(f"\nTotal operations with counter data: {total_ops}")

    # Get all unique counter types
    counter_types = sorted(perf_counter_df["counter type"].unique())

    print("\n" + "=" * 100)
    print("RAW COUNTER VALUES")
    print("=" * 100)
    print(f"{'Counter Type':<40} {'Statistic':<12} {'Ops':>8} {'Min':>15} {'Median':>15} {'Max':>15} {'Avg':>15}")
    print("-" * 100)

    for counter_type in counter_types:
        counter_data = perf_counter_df[perf_counter_df["counter type"] == counter_type]
        counter_grouped = counter_data.groupby(["run_host_id", "trace_id_count"])

        # Calculate statistics across operations
        min_vals = counter_grouped["value"].min()
        median_vals = counter_grouped["value"].median()
        max_vals = counter_grouped["value"].max()
        avg_vals = counter_grouped["value"].mean()

        ops_with_data = len(counter_grouped)

        # Print value statistics
        print(
            f"{counter_type:<40} {'Value':<12} {ops_with_data:>8} "
            f"{min_vals.min():>15.1f} {median_vals.median():>15.1f} "
            f"{max_vals.max():>15.1f} {avg_vals.mean():>15.1f}"
        )

    print("\n" + "=" * 100)
    print("COUNTER REFERENCE COUNTS")
    print("=" * 100)
    print(f"{'Counter Type':<40} {'Statistic':<12} {'Ops':>8} {'Min':>15} {'Median':>15} {'Max':>15} {'Avg':>15}")
    print("-" * 100)

    for counter_type in counter_types:
        counter_data = perf_counter_df[perf_counter_df["counter type"] == counter_type]
        counter_grouped = counter_data.groupby(["run_host_id", "trace_id_count"])

        # Calculate statistics for reference counts
        min_refs = counter_grouped["ref cnt"].min()
        median_refs = counter_grouped["ref cnt"].median()
        max_refs = counter_grouped["ref cnt"].max()
        avg_refs = counter_grouped["ref cnt"].mean()

        ops_with_data = len(counter_grouped)

        # Print ref count statistics
        print(
            f"{counter_type:<40} {'Ref Count':<12} {ops_with_data:>8} "
            f"{min_refs.min():>15.1f} {median_refs.median():>15.1f} "
            f"{max_refs.max():>15.1f} {avg_refs.mean():>15.1f}"
        )

    print("\n" + "=" * 100)
    print("COUNTER UTILIZATION (%)")
    print("=" * 100)
    print(f"{'Counter Type':<40} {'Statistic':<12} {'Ops':>8} {'Min':>15} {'Median':>15} {'Max':>15} {'Avg':>15}")
    print("-" * 100)

    for counter_type in counter_types:
        counter_data = perf_counter_df[perf_counter_df["counter type"] == counter_type].copy()
        counter_data["util"] = (counter_data["value"] / counter_data["ref cnt"] * 100).replace(
            [float("inf"), -float("inf")], nan
        )
        counter_grouped = counter_data.groupby(["run_host_id", "trace_id_count"])

        # Calculate utilization statistics
        min_utils = counter_grouped["util"].min()
        median_utils = counter_grouped["util"].median()
        max_utils = counter_grouped["util"].max()
        avg_utils = counter_grouped["util"].mean()

        ops_with_data = len(counter_grouped)

        # Print utilization statistics
        print(
            f"{counter_type:<40} {'Utilization':<12} {ops_with_data:>8} "
            f"{min_utils.min():>14.2f}% {median_utils.median():>14.2f}% "
            f"{max_utils.max():>14.2f}% {avg_utils.mean():>14.2f}%"
        )

    print("\n" + "=" * 100 + "\n")


def print_efficiency_metrics_summary(metrics_df: pd.DataFrame, device_id: int) -> None:
    """Print a summary of calculated efficiency metrics grouped by metric type."""
    if metrics_df is None or metrics_df.empty:
        return

    print("\n" + "=" * 100)
    print(f"EFFICIENCY METRICS SUMMARY - DEVICE {device_id}")
    print("=" * 100)

    print(f"\nTotal operations with metrics: {len(metrics_df)}")

    # Define metric names (without the Min/Median/Max/Avg suffix)
    base_metrics = [
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
        # INSTRN_THREAD metrics
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
        # TDMA_UNPACK
        "Data Hazard Stall Rate",
        # L1 Bank 0
        "L1 Unpacker Port Util",
        "L1 TDMA Bundle Util",
        "NOC Ring 0 Outgoing Util",
        "NOC Ring 0 Incoming Util",
        # L1 Bank 1
        "NOC Ring 1 Outgoing Util",
        "NOC Ring 1 Incoming Util",
        # L1 Port 1 (arch-specific: BH unified packer, WH unpacker#1/ECC/pack1)
        "L1 Packer Port Util",
        # L1 back-pressure (derived stall: (req - grant) / req)
        "NOC Ring 0 Outgoing Backpressure",
        "NOC Ring 0 Incoming Backpressure",
        "NOC Ring 1 Outgoing Backpressure",
        "NOC Ring 1 Incoming Backpressure",
        "L1 Unpacker Backpressure",
        "L1 Packer Port Backpressure",
        # Fidelity and math pipeline stall breakdown
        "HiFi2 Instrn Rate",
        "LoFi Instrn Rate",
        "Math Src Data Ready Rate",
        "SrcA Write Port Blocked Rate",
        "Dest Read Backpressure",
        "Math Dest Write Port Stall Rate",
        "Math Scoreboard Stall Rate",
    ]

    # Non-percentage metrics (raw rates, not %)
    base_metrics_no_pct = [
        "Thread 0 IPC",
        "Thread 1 IPC",
        "Thread 2 IPC",
        "Unpack Instrn Issue Rate T0",
        "Math Instrn Issue Rate T1",
        "Pack Instrn Issue Rate T2",
    ]

    # For each base metric, display a table with Min/Median/Max/Avg rows
    for base_metric in base_metrics + base_metrics_no_pct:
        is_pct = base_metric not in base_metrics_no_pct
        suffix = " (%)" if is_pct else ""
        unit = "%" if is_pct else ""

        print("\n" + "=" * 80)
        print(f"{base_metric.upper()}")
        print("=" * 80)

        # Create table header
        print(f"{'Statistic':<12} {'Ops with Data':>15} {'Range':>30} {'Mean':>12}")
        print("-" * 80)

        # Check each statistic
        total_ops = len(metrics_df)
        for stat in ["Min", "Median", "Max", "Avg"]:
            col_name = f"{base_metric} {stat}{suffix}"
            if col_name in metrics_df.columns:
                non_nan = metrics_df[col_name].dropna()
                if len(non_nan) > 0:
                    ops_with_data = f"{len(non_nan)}/{total_ops}"
                    range_str = f"{non_nan.min():.2f}{unit} - {non_nan.max():.2f}{unit}"
                    mean_str = f"{non_nan.mean():.2f}{unit}"
                else:
                    ops_with_data = f"0/{total_ops}"
                    range_str = "N/A"
                    mean_str = "N/A"

                print(f"{stat:<12} {ops_with_data:>15} {range_str:>30} {mean_str:>12}")

    print("\n" + "=" * 100 + "\n")


def get_device_op_data(ops: Dict[int, OpDict], host_device_op_compare) -> Tuple[DeviceOpsDict, bool]:
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
