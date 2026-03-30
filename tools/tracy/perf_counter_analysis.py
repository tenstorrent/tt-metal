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

# Counter type enum from perf_counters.hpp (IDs 0..94: 95 entries including UNDEF=0)
COUNTER_TYPE_NAMES = {
    0: "UNDEF",
    # FPU Group (3)
    1: "FPU_COUNTER",
    2: "SFPU_COUNTER",
    3: "MATH_COUNTER",
    # TDMA_UNPACK Group (11)
    4: "DATA_HAZARD_STALLS_MOVD2A",
    5: "MATH_INSTRN_STARTED",
    6: "MATH_INSTRN_AVAILABLE",
    7: "SRCB_WRITE_AVAILABLE",
    8: "SRCA_WRITE_AVAILABLE",
    9: "UNPACK0_BUSY_THREAD0",
    10: "UNPACK1_BUSY_THREAD0",
    11: "UNPACK0_BUSY_THREAD1",
    12: "UNPACK1_BUSY_THREAD1",
    13: "SRCB_WRITE",
    14: "SRCA_WRITE",
    # TDMA_PACK Group (3)
    15: "PACKER_DEST_READ_AVAILABLE",
    16: "PACKER_BUSY",
    17: "AVAILABLE_MATH",
    # INSTRN_THREAD Group (61)
    18: "CFG_INSTRN_AVAILABLE_0",
    19: "CFG_INSTRN_AVAILABLE_1",
    20: "CFG_INSTRN_AVAILABLE_2",
    21: "SYNC_INSTRN_AVAILABLE_0",
    22: "SYNC_INSTRN_AVAILABLE_1",
    23: "SYNC_INSTRN_AVAILABLE_2",
    24: "THCON_INSTRN_AVAILABLE_0",
    25: "THCON_INSTRN_AVAILABLE_1",
    26: "THCON_INSTRN_AVAILABLE_2",
    27: "XSEARCH_INSTRN_AVAILABLE_0",
    28: "XSEARCH_INSTRN_AVAILABLE_1",
    29: "XSEARCH_INSTRN_AVAILABLE_2",
    30: "MOVE_INSTRN_AVAILABLE_0",
    31: "MOVE_INSTRN_AVAILABLE_1",
    32: "MOVE_INSTRN_AVAILABLE_2",
    33: "FPU_INSTRN_AVAILABLE_0",
    34: "FPU_INSTRN_AVAILABLE_1",
    35: "FPU_INSTRN_AVAILABLE_2",
    36: "UNPACK_INSTRN_AVAILABLE_0",
    37: "UNPACK_INSTRN_AVAILABLE_1",
    38: "UNPACK_INSTRN_AVAILABLE_2",
    39: "PACK_INSTRN_AVAILABLE_0",
    40: "PACK_INSTRN_AVAILABLE_1",
    41: "PACK_INSTRN_AVAILABLE_2",
    42: "THREAD_STALLS_0",
    43: "THREAD_STALLS_1",
    44: "THREAD_STALLS_2",
    45: "WAITING_FOR_SRCA_CLEAR",
    46: "WAITING_FOR_SRCB_CLEAR",
    47: "WAITING_FOR_SRCA_VALID",
    48: "WAITING_FOR_SRCB_VALID",
    49: "WAITING_FOR_THCON_IDLE_0",
    50: "WAITING_FOR_THCON_IDLE_1",
    51: "WAITING_FOR_THCON_IDLE_2",
    52: "WAITING_FOR_UNPACK_IDLE_0",
    53: "WAITING_FOR_UNPACK_IDLE_1",
    54: "WAITING_FOR_UNPACK_IDLE_2",
    55: "WAITING_FOR_PACK_IDLE_0",
    56: "WAITING_FOR_PACK_IDLE_1",
    57: "WAITING_FOR_PACK_IDLE_2",
    58: "WAITING_FOR_MATH_IDLE_0",
    59: "WAITING_FOR_MATH_IDLE_1",
    60: "WAITING_FOR_MATH_IDLE_2",
    61: "WAITING_FOR_NONZERO_SEM_0",
    62: "WAITING_FOR_NONZERO_SEM_1",
    63: "WAITING_FOR_NONZERO_SEM_2",
    64: "WAITING_FOR_NONFULL_SEM_0",
    65: "WAITING_FOR_NONFULL_SEM_1",
    66: "WAITING_FOR_NONFULL_SEM_2",
    67: "WAITING_FOR_MOVE_IDLE_0",
    68: "WAITING_FOR_MOVE_IDLE_1",
    69: "WAITING_FOR_MOVE_IDLE_2",
    70: "WAITING_FOR_MMIO_IDLE_0",
    71: "WAITING_FOR_MMIO_IDLE_1",
    72: "WAITING_FOR_MMIO_IDLE_2",
    73: "WAITING_FOR_SFPU_IDLE_0",
    74: "WAITING_FOR_SFPU_IDLE_1",
    75: "WAITING_FOR_SFPU_IDLE_2",
    76: "THREAD_INSTRUCTIONS_0",
    77: "THREAD_INSTRUCTIONS_1",
    78: "THREAD_INSTRUCTIONS_2",
    # L1 Bank 0 (MUX_CTRL bit 4 = 0, monitors L1 ports 0-7)
    79: "L1_0_UNPACKER_0",
    80: "L1_0_UNPACKER_1_ECC_PACK1",
    81: "L1_0_TDMA_BUNDLE_0_RISC",
    82: "L1_0_TDMA_BUNDLE_1_TRISC",
    83: "L1_0_NOC_RING0_OUTGOING_0",
    84: "L1_0_NOC_RING0_OUTGOING_1",
    85: "L1_0_NOC_RING0_INCOMING_0",
    86: "L1_0_NOC_RING0_INCOMING_1",
    # L1 Bank 1 (MUX_CTRL bit 4 = 1, monitors L1 ports 8-15)
    87: "L1_1_TDMA_PACKER_2",
    88: "L1_1_EXT_UNPACKER_1",
    89: "L1_1_EXT_UNPACKER_2",
    90: "L1_1_EXT_UNPACKER_3",
    91: "L1_1_NOC_RING1_OUTGOING_0",
    92: "L1_1_NOC_RING1_OUTGOING_1",
    93: "L1_1_NOC_RING1_INCOMING_0",
    94: "L1_1_NOC_RING1_INCOMING_1",
    # Blackhole-specific L1 ports (differ from Wormhole at ports 1 and 8)
    95: "L1_0_UNIFIED_PACKER",  # BH Port 1, mux 0: Unified Packer
    96: "L1_1_RISC_CORE",  # BH Port 8, mux 1: RISC Core L1 access
    # INSTRN_THREAD grant counters: instruction issue counts (8 types x 3 threads)
    97: "CFG_INSTRN_ISSUED_0",
    98: "CFG_INSTRN_ISSUED_1",
    99: "CFG_INSTRN_ISSUED_2",
    100: "SYNC_INSTRN_ISSUED_0",
    101: "SYNC_INSTRN_ISSUED_1",
    102: "SYNC_INSTRN_ISSUED_2",
    103: "THCON_INSTRN_ISSUED_0",
    104: "THCON_INSTRN_ISSUED_1",
    105: "THCON_INSTRN_ISSUED_2",
    106: "XSEARCH_INSTRN_ISSUED_0",
    107: "XSEARCH_INSTRN_ISSUED_1",
    108: "XSEARCH_INSTRN_ISSUED_2",
    109: "MOVE_INSTRN_ISSUED_0",
    110: "MOVE_INSTRN_ISSUED_1",
    111: "MOVE_INSTRN_ISSUED_2",
    112: "FPU_INSTRN_ISSUED_0",
    113: "FPU_INSTRN_ISSUED_1",
    114: "FPU_INSTRN_ISSUED_2",
    115: "UNPACK_INSTRN_ISSUED_0",
    116: "UNPACK_INSTRN_ISSUED_1",
    117: "UNPACK_INSTRN_ISSUED_2",
    118: "PACK_INSTRN_ISSUED_0",
    119: "PACK_INSTRN_ISSUED_1",
    120: "PACK_INSTRN_ISSUED_2",
    # TDMA_UNPACK grant counters
    121: "INSTRN_2_HF_CYCLES",
    122: "INSTRN_1_HF_CYCLE",
    123: "SRCB_WRITE_ACTUAL",
    124: "SRCA_WRITE_NOT_BLOCKED_OVR",
    125: "SRCA_WRITE_ACTUAL",
    126: "SRCB_WRITE_NOT_BLOCKED_PORT",
    127: "SRCA_WRITE_THREAD0",
    128: "SRCB_WRITE_THREAD0",
    129: "SRCA_WRITE_THREAD1",
    130: "SRCB_WRITE_THREAD1",
    131: "MATH_INSTRN_NOT_BLOCKED_SRC",
    # TDMA_PACK additional req counters
    132: "PACKER_DEST_READ_1",
    133: "PACKER_DEST_READ_2",
    134: "PACKER_DEST_READ_3",
    135: "PACKER_BUSY_0",
    136: "PACKER_BUSY_1",
    137: "PACKER_BUSY_2",
    # TDMA_PACK grant counters
    138: "DEST_READ_GRANTED_0",
    139: "DEST_READ_GRANTED_1",
    140: "DEST_READ_GRANTED_2",
    141: "DEST_READ_GRANTED_3",
    142: "MATH_NOT_STALLED_DEST_WR_PORT",
    # L1 grant counters (reqif_ready — cycles L1 was ready to accept)
    # L1 Bank 0 grants
    143: "L1_0_UNPACKER_0_GRANT",
    144: "L1_0_PORT1_GRANT",
    145: "L1_0_TDMA_BUNDLE_0_GRANT",
    146: "L1_0_TDMA_BUNDLE_1_GRANT",
    147: "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    148: "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    149: "L1_0_NOC_RING0_INCOMING_0_GRANT",
    150: "L1_0_NOC_RING0_INCOMING_1_GRANT",
    # L1 Bank 1 grants
    151: "L1_1_PORT8_GRANT",
    152: "L1_1_EXT_UNPACKER_1_GRANT",
    153: "L1_1_EXT_UNPACKER_2_GRANT",
    154: "L1_1_EXT_UNPACKER_3_GRANT",
    155: "L1_1_NOC_RING1_OUTGOING_0_GRANT",
    156: "L1_1_NOC_RING1_OUTGOING_1_GRANT",
    157: "L1_1_NOC_RING1_INCOMING_0_GRANT",
    158: "L1_1_NOC_RING1_INCOMING_1_GRANT",
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
        # L1 back-pressure
        "NOC Ring 0 Outgoing Backpressure",
        "NOC Ring 0 Incoming Backpressure",
    ]

    # IPC metrics use no suffix (not percentages)
    base_metrics_no_pct = [
        "Thread 0 IPC",
        "Thread 1 IPC",
        "Thread 2 IPC",
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
