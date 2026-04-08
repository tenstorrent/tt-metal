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

# BH RTL-confirmed dead counters: signals that never produce useful data on BH silicon
# (verified across 8 diverse workloads). Note: some RTL signals appear live in the RTL
# source but are empirically dead on silicon (MATH_INSTRN_STARTED, SFPU_IDLE), and some
# RTL signals appear dead but are empirically live (PACKER_DEST_READ_1, DEST_READ_GRANTED_1).
# This list reflects the empirical truth, not just RTL analysis.
# - PACK banks 2-3 req/grant: tied to 1'b0 (PACK_COUNT=1, only 1 packer engine)
# - PACKER_BUSY_0/1/2: individual packer engine busy, tied to 1'b0 (PACK_COUNT=1)
# - PACK_BANK6/7_GRANT: grant vector bits tied to 2'b00
# - FIDELITY_PHASE_STALLS: fidelity_phases_ongoing = 1'b0 on BH (same as WH)
# - MATH_INSTRN_STARTED: o_math_instrnbuf_rden inactive on BH silicon (despite RTL showing live)
# - HF cycle counters: gated by hf_cycles which is always 0 (fidelity_phases_ongoing = 1'b0)
# - SFPU_IDLE counters: empirically 0 on BH silicon across all workloads including SFPU-heavy
BH_RTL_DEAD_COUNTERS = frozenset(
    {
        "PACKER_DEST_READ_2",
        "PACKER_DEST_READ_3",
        "PACKER_BUSY_0",
        "PACKER_BUSY_1",
        "PACKER_BUSY_2",
        "DEST_READ_GRANTED_2",
        "DEST_READ_GRANTED_3",
        "PACK_BANK6_GRANT",
        "PACK_BANK7_GRANT",
        "FIDELITY_PHASE_STALLS",
        "MATH_INSTRN_STARTED",
        "MATH_INSTRN_NOT_BLOCKED_SRC",
        "INSTRN_2_HF_CYCLES",
        "INSTRN_1_HF_CYCLE",
        "WAITING_FOR_SFPU_IDLE_0",
        "WAITING_FOR_SFPU_IDLE_1",
        "WAITING_FOR_SFPU_IDLE_2",
    }
)

# WH RTL-confirmed dead counters: signals tied to constant 0 in Wormhole RTL.
# - PACK_BANK6/7_GRANT: bank 6-7 grant tied to 2'b00 (same as BH)
# - FIDELITY_PHASE_STALLS: fidelity_phases_ongoing = 1'b0 on WH (no multi-phase fidelity)
WH_RTL_DEAD_COUNTERS = frozenset(
    {
        "PACK_BANK6_GRANT",
        "PACK_BANK7_GRANT",
        "FIDELITY_PHASE_STALLS",
    }
)

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
    # L1 Bank 2 (BH only, mux position 2, NOC Ring 2 ports 16-23)
    177: "L1_2_NOC_RING2_PORT_0",
    178: "L1_2_NOC_RING2_PORT_1",
    179: "L1_2_NOC_RING2_PORT_2",
    180: "L1_2_NOC_RING2_PORT_3",
    181: "L1_2_NOC_RING2_PORT_4",
    182: "L1_2_NOC_RING2_PORT_5",
    183: "L1_2_NOC_RING2_PORT_6",
    184: "L1_2_NOC_RING2_PORT_7",
    185: "L1_2_NOC_RING2_PORT_0_GRANT",
    186: "L1_2_NOC_RING2_PORT_1_GRANT",
    187: "L1_2_NOC_RING2_PORT_2_GRANT",
    188: "L1_2_NOC_RING2_PORT_3_GRANT",
    189: "L1_2_NOC_RING2_PORT_4_GRANT",
    190: "L1_2_NOC_RING2_PORT_5_GRANT",
    191: "L1_2_NOC_RING2_PORT_6_GRANT",
    192: "L1_2_NOC_RING2_PORT_7_GRANT",
    # L1 Bank 3 (BH only, mux position 3, NOC Ring 3 ports 24-31)
    193: "L1_3_NOC_RING3_PORT_0",
    194: "L1_3_NOC_RING3_PORT_1",
    195: "L1_3_NOC_RING3_PORT_2",
    196: "L1_3_NOC_RING3_PORT_3",
    197: "L1_3_NOC_RING3_PORT_4",
    198: "L1_3_NOC_RING3_PORT_5",
    199: "L1_3_NOC_RING3_PORT_6",
    200: "L1_3_NOC_RING3_PORT_7",
    201: "L1_3_NOC_RING3_PORT_0_GRANT",
    202: "L1_3_NOC_RING3_PORT_1_GRANT",
    203: "L1_3_NOC_RING3_PORT_2_GRANT",
    204: "L1_3_NOC_RING3_PORT_3_GRANT",
    205: "L1_3_NOC_RING3_PORT_4_GRANT",
    206: "L1_3_NOC_RING3_PORT_5_GRANT",
    207: "L1_3_NOC_RING3_PORT_6_GRANT",
    208: "L1_3_NOC_RING3_PORT_7_GRANT",
}


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
    # Thread IPC removed: no RTL counter for instruction counts (sel 24-26 = total stalls)
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


def extract_perf_counters(events: List[Any], arch: str = "") -> Optional[pd.DataFrame]:
    # If perf counter data exists, extract relevant columns and return as a dataframe
    EVENT_METADATA_IDX = 0
    EVENT_TIMESTAMP_IDX = 1
    EVENT_RISC_TYPE_IDX = 3
    EVENT_CORE_COORDS_IDX = 4
    PERF_COUNTER_ID = 9090

    arch_lower = arch.lower() if arch else ""
    dead_counters = frozenset()
    if arch_lower == "blackhole":
        dead_counters = BH_RTL_DEAD_COUNTERS
    elif arch_lower in ("wormhole", "wormhole_b0"):
        dead_counters = WH_RTL_DEAD_COUNTERS

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

                # Skip RTL-confirmed dead counters (signals tied to 0 in silicon)
                if dead_counters and counter_type_name in dead_counters:
                    continue

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
    pct_metrics = [
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
        # NEW: Per-type instruction issue efficiency
        "CFG Instrn Avail Rate T0",
        "SYNC Instrn Avail Rate T0",
        "THCON Instrn Avail Rate T0",
        "MOVE Instrn Avail Rate T0",
        "MATH Instrn Avail Rate T1",
        "UNPACK Instrn Avail Rate T0",
        "PACK Instrn Avail Rate T2",
        # NEW: Write port blocking
        "SrcB Write Port Blocked Rate",
        "SrcA Write Actual Efficiency",
        "SrcB Write Actual Efficiency",
        # NEW: Fidelity analysis
        "HiFi4 Instrn Rate",
        "Fidelity Phase Overhead",
        # NEW: Packer engine granularity
        "Packer Engine 0 Util",
        "Packer Engine 1 Util",
        "Packer Engine 2 Util",
        # NEW: Low priority waits
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

    # Non-percentage metrics (raw rates, not %)
    rate_metrics = [
        # Thread IPC removed: no RTL counter for instruction counts
        "Unpack Instrn Issue Rate T0",
        "Math Instrn Issue Rate T1",
        "Pack Instrn Issue Rate T2",
    ]

    # For each base metric, display a table with Min/Median/Max/Avg rows
    for base_metric in pct_metrics + rate_metrics:
        is_pct = base_metric not in rate_metrics
        suffix = " (%)" if is_pct else ""
        unit = "%" if is_pct else ""

        # Skip metrics that have no data columns (e.g. BH-dead counters)
        avg_col = f"{base_metric} Avg{suffix}"
        if avg_col not in metrics_df.columns or metrics_df[avg_col].dropna().empty:
            continue

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


def compute_perf_counter_metrics(perf_counter_df, device_arch, total_compute_cores):
    """Compute all perf counter metrics from the perf_counter_df DataFrame.

    Parameters
    ----------
    perf_counter_df : pd.DataFrame
        DataFrame with columns including "counter type", "value", "ref cnt",
        "run_host_id", "trace_id_count", "core_x", "core_y".
    device_arch : str
        Device architecture string (e.g. "wormhole_b0", "blackhole").
    total_compute_cores : int
        Number of compute cores on the device.

    Returns
    -------
    dict with two keys:
        "per_op_stats" : dict mapping metric_base_name -> {"min": dict, "median": dict, "max": dict, "avg": dict}
        "per_op_counts" : dict mapping count_name -> dict
    """

    per_op_stats = {}
    per_op_counts = {}

    # ---- Helper functions ----

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

    def _group_to_stat_dict(series):
        """Group a per-core series by op and return min/median/max/avg dicts."""
        grouped = series.groupby(level=["run_host_id", "trace_id_count"])
        return {
            "min": grouped.min().to_dict(),
            "median": grouped.median().to_dict(),
            "max": grouped.max().to_dict(),
            "avg": grouped.mean().to_dict(),
        }

    # ---- Metric computation (moved from process_ops_logs.py) ----

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
    per_op_stats["SFPU Util"] = _group_to_stat_dict(sfpu_util)
    per_op_counts["avg_sfpu_count"] = (
        sfpu_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
    ).to_dict()

    # FPU Counter aggregations
    per_op_stats["FPU Util"] = _group_to_stat_dict(fpu_util)
    per_op_counts["avg_fpu_count"] = (
        fpu_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
    ).to_dict()

    # MATH Counter aggregations
    per_op_stats["MATH Util"] = _group_to_stat_dict(math_util)
    per_op_counts["avg_math_count"] = (
        math_counter.groupby(level=["run_host_id", "trace_id_count"]).sum() / total_compute_cores
    ).to_dict()

    # Calculate per-core efficiency metrics
    unpack0_eff = (srca_write / unpack0_busy * 100).replace([float("inf"), -float("inf")], nan)
    unpack1_eff = (srcb_write / unpack1_busy * 100).replace([float("inf"), -float("inf")], nan)

    # Packer Efficiency: On WH, PACKER_DEST_READ_AVAILABLE / PACKER_BUSY.
    # On BH, PACKER_BUSY is always 0 (packer completes within gated clock window).
    # Fallback: use DEST_READ_GRANTED_1 / PACKER_DEST_READ_AVAILABLE which measures
    # what fraction of dest read requests were granted (same signal on silicon,
    # confirmed both read 3968 on matmul = 0% backpressure = 100% efficiency).
    if packer_busy is not None and packer_busy.sum() > 0:
        pack_eff = (packer_dest_read / packer_busy * 100).replace([float("inf"), -float("inf")], nan)
    elif has_counter("DEST_READ_GRANTED_1"):
        dest_granted_1 = get_counter_series("DEST_READ_GRANTED_1")
        pack_eff = (dest_granted_1 / packer_dest_read * 100).replace([float("inf"), -float("inf")], nan)
    else:
        pack_eff = pd.Series(dtype=float)

    # Math Pipeline Utilization: On BH, MATH_INSTRN_STARTED is always 0 (RTL dead).
    # Fall back to FIDELITY_PHASE_STALLS / ref_cnt (compute_util_metric) which measures
    # what % of time the math pipeline was actively executing (including HiFi phases).
    if math_instrn_started is not None and math_instrn_started.sum() > 0:
        math_pipe_util = (math_instrn_started / math_instrn_available * 100).replace([float("inf"), -float("inf")], nan)
    else:
        # Use FIDELITY_PHASE_STALLS as a proxy: it counts cycles where
        # math_instrn_valid & fidelity_phases_ongoing, i.e. math pipeline active.
        fidelity = get_counter_series("FIDELITY_PHASE_STALLS") if has_counter("FIDELITY_PHASE_STALLS") else None
        fidelity_ref = get_counter_ref_cnt("FIDELITY_PHASE_STALLS") if has_counter("FIDELITY_PHASE_STALLS") else None
        if fidelity is not None and fidelity_ref is not None:
            math_pipe_util = (fidelity / fidelity_ref * 100).replace([float("inf"), -float("inf")], nan)
        else:
            math_pipe_util = pd.Series(dtype=float)

    # Math-to-Pack Handoff Efficiency: On BH, PACKER_BUSY is always 0.
    # Fall back to AVAILABLE_MATH / ref_cnt (% of time math not stalled by scoreboard).
    if packer_busy is not None and packer_busy.sum() > 0:
        math_pack_eff = (available_math / packer_busy * 100).replace([float("inf"), -float("inf")], nan)
    elif available_math is not None:
        avail_ref = get_counter_ref_cnt("AVAILABLE_MATH") if has_counter("AVAILABLE_MATH") else None
        if avail_ref is not None:
            math_pack_eff = (available_math / avail_ref * 100).replace([float("inf"), -float("inf")], nan)
        else:
            math_pack_eff = pd.Series(dtype=float)
    else:
        math_pack_eff = pd.Series(dtype=float)

    unpack_math_flow = (
        ((srca_write_avail + srcb_write_avail) / 2) / ((unpack0_busy + unpack1_busy) / 2) * 100
    ).replace([float("inf"), -float("inf")], nan)

    # Aggregate per operation (min, median, max, avg)
    per_op_stats["Unpacker0 Write Efficiency"] = _group_to_stat_dict(unpack0_eff)
    per_op_stats["Unpacker1 Write Efficiency"] = _group_to_stat_dict(unpack1_eff)

    # Combined Unpacker Write Efficiency (average per core, then aggregate)
    unpack_combined = pd.concat([unpack0_eff, unpack1_eff], axis=1).mean(axis=1, skipna=True)
    per_op_stats["Unpacker Write Efficiency"] = _group_to_stat_dict(unpack_combined)

    per_op_stats["Packer Efficiency"] = _group_to_stat_dict(pack_eff)

    # FPU Execution Efficiency: FPU_COUNTER / FPU_INSTRN_AVAILABLE_1
    fpu_exec_eff = (fpu_counter / fpu_instrn_available_1 * 100).replace([float("inf"), -float("inf")], nan)
    per_op_stats["FPU Execution Efficiency"] = _group_to_stat_dict(fpu_exec_eff)

    # Math Pipeline Utilization
    per_op_stats["Math Pipeline Utilization"] = _group_to_stat_dict(math_pipe_util)

    # Math-to-Pack Handoff Efficiency
    per_op_stats["Math-to-Pack Handoff Efficiency"] = _group_to_stat_dict(math_pack_eff)

    # Unpacker-to-Math Data Flow
    per_op_stats["Unpacker-to-Math Data Flow"] = _group_to_stat_dict(unpack_math_flow)

    # === Thread stall metrics ===
    for t in range(3):
        name = f"THREAD_STALLS_{t}"
        if has_counter(name):
            per_op_stats[f"Thread {t} Stall Rate"] = compute_util_metric(name)

    # Thread IPC: no RTL counter for instruction counts. IPC metric removed.

    # Pipeline wait metrics
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
            per_op_stats[metric_name] = compute_util_metric(counter_name)

    # Semaphore wait metrics
    for t in range(3):
        zero_name = f"WAITING_FOR_NONZERO_SEM_{t}"
        full_name = f"WAITING_FOR_NONFULL_SEM_{t}"
        if has_counter(zero_name):
            per_op_stats[f"Semaphore Zero Wait T{t}"] = compute_util_metric(zero_name)
        if has_counter(full_name):
            per_op_stats[f"Semaphore Full Wait T{t}"] = compute_util_metric(full_name)

    # === TDMA_UNPACK data hazard ===
    if has_counter("DATA_HAZARD_STALLS_MOVD2A"):
        per_op_stats["Data Hazard Stall Rate"] = compute_util_metric("DATA_HAZARD_STALLS_MOVD2A")

    # === L1 Bank 0 ===
    if has_counter("L1_0_UNPACKER_0"):
        per_op_stats["L1 Unpacker Port Util"] = compute_util_metric("L1_0_UNPACKER_0")
    if has_counter("L1_0_TDMA_BUNDLE_0_RISC") and has_counter("L1_0_TDMA_BUNDLE_1_TRISC"):
        per_op_stats["L1 TDMA Bundle Util"] = compute_avg_channel_util(
            "L1_0_TDMA_BUNDLE_0_RISC", "L1_0_TDMA_BUNDLE_1_TRISC"
        )
    if has_counter("L1_0_NOC_RING0_OUTGOING_0") and has_counter("L1_0_NOC_RING0_OUTGOING_1"):
        per_op_stats["NOC Ring 0 Outgoing Util"] = compute_avg_channel_util(
            "L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1"
        )
    if has_counter("L1_0_NOC_RING0_INCOMING_0") and has_counter("L1_0_NOC_RING0_INCOMING_1"):
        per_op_stats["NOC Ring 0 Incoming Util"] = compute_avg_channel_util(
            "L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1"
        )

    # L1 Port 1 (arch-specific: BH unified packer or WH unpacker#1/ECC/pack1)
    if has_counter("L1_0_UNIFIED_PACKER"):
        per_op_stats["L1 Packer Port Util"] = compute_util_metric("L1_0_UNIFIED_PACKER")
    elif has_counter("L1_0_UNPACKER_1_ECC_PACK1"):
        per_op_stats["L1 Packer Port Util"] = compute_util_metric("L1_0_UNPACKER_1_ECC_PACK1")

    # L1 back-pressure metrics (from grant counters)
    if has_counter("L1_0_NOC_RING0_OUTGOING_0") and has_counter("L1_0_NOC_RING0_OUTGOING_0_GRANT"):
        per_op_stats["NOC Ring 0 Outgoing Backpressure"] = compute_backpressure(
            "L1_0_NOC_RING0_OUTGOING_0",
            "L1_0_NOC_RING0_OUTGOING_1",
            "L1_0_NOC_RING0_OUTGOING_0_GRANT",
            "L1_0_NOC_RING0_OUTGOING_1_GRANT",
        )
    if has_counter("L1_0_NOC_RING0_INCOMING_0") and has_counter("L1_0_NOC_RING0_INCOMING_0_GRANT"):
        per_op_stats["NOC Ring 0 Incoming Backpressure"] = compute_backpressure(
            "L1_0_NOC_RING0_INCOMING_0",
            "L1_0_NOC_RING0_INCOMING_1",
            "L1_0_NOC_RING0_INCOMING_0_GRANT",
            "L1_0_NOC_RING0_INCOMING_1_GRANT",
        )

    # === Grant counter derived metrics ===
    # Fidelity cycle breakdown
    if has_counter("INSTRN_2_HF_CYCLES") and has_counter("MATH_INSTRN_STARTED"):
        num = get_counter_series("INSTRN_2_HF_CYCLES")
        den = get_counter_series("MATH_INSTRN_STARTED")
        ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["HiFi2 Instrn Rate"] = _group_to_stat_dict(ratio)
    if has_counter("INSTRN_1_HF_CYCLE") and has_counter("MATH_INSTRN_STARTED"):
        num = get_counter_series("INSTRN_1_HF_CYCLE")
        den = get_counter_series("MATH_INSTRN_STARTED")
        ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["LoFi Instrn Rate"] = _group_to_stat_dict(ratio)

    # Math source data readiness
    # On WH, MATH_INSTRN_NOT_BLOCKED_SRC (counter_sel 256) measures 4-HF-cycle instructions,
    # not math-blocked-by-src. Only compute this metric on BH where it's the correct signal.
    is_wh = device_arch.lower() in ("wormhole", "wormhole_b0")
    if not is_wh and has_counter("MATH_INSTRN_NOT_BLOCKED_SRC") and has_counter("MATH_INSTRN_AVAILABLE"):
        num = get_counter_series("MATH_INSTRN_NOT_BLOCKED_SRC")
        den = get_counter_series("MATH_INSTRN_AVAILABLE")
        ratio = (num / den * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["Math Src Data Ready Rate"] = _group_to_stat_dict(ratio)

    # SrcA write port blocked rate
    if has_counter("SRCA_WRITE_AVAILABLE") and has_counter("SRCA_WRITE_NOT_BLOCKED_OVR"):
        avail = get_counter_series("SRCA_WRITE_AVAILABLE")
        unblocked = get_counter_series("SRCA_WRITE_NOT_BLOCKED_OVR")
        ratio = ((avail - unblocked) / avail * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["SrcA Write Port Blocked Rate"] = _group_to_stat_dict(ratio)

    # Dest read backpressure
    dest_grant_name = "DEST_READ_GRANTED_1" if has_counter("DEST_READ_GRANTED_1") else "DEST_READ_GRANTED_0"
    if has_counter("PACKER_DEST_READ_AVAILABLE") and has_counter(dest_grant_name):
        req = get_counter_series("PACKER_DEST_READ_AVAILABLE")
        grant = get_counter_series(dest_grant_name)
        ratio = ((req - grant) / req * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["Dest Read Backpressure"] = _group_to_stat_dict(ratio)

    # Math dest write port stall and scoreboard stall
    if has_counter("MATH_INSTRN_AVAILABLE") and has_counter("MATH_NOT_STALLED_DEST_WR_PORT"):
        unstalled = get_counter_series("MATH_NOT_STALLED_DEST_WR_PORT")
        # On BH, MATH_NOT_STALLED_DEST_WR_PORT is always 0 (RTL dead), producing
        # a bogus 100% stall rate. Only compute when the counter has real data.
        if unstalled.sum() > 0:
            avail = get_counter_series("MATH_INSTRN_AVAILABLE")
            ratio = ((avail - unstalled) / avail * 100).replace([float("inf"), -float("inf")], nan)
            per_op_stats["Math Dest Write Port Stall Rate"] = _group_to_stat_dict(ratio)
    if has_counter("MATH_INSTRN_AVAILABLE") and has_counter("AVAILABLE_MATH"):
        avail = get_counter_series("MATH_INSTRN_AVAILABLE")
        unstalled = get_counter_series("AVAILABLE_MATH")
        ratio = ((avail - unstalled) / avail * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["Math Scoreboard Stall Rate"] = _group_to_stat_dict(ratio)

    # Instruction issue rates per thread
    if has_counter("UNPACK_INSTRN_ISSUED_0"):
        per_op_stats["Unpack Instrn Issue Rate T0"] = compute_util_metric("UNPACK_INSTRN_ISSUED_0", scale=1)
    if has_counter("FPU_INSTRN_ISSUED_1"):
        per_op_stats["Math Instrn Issue Rate T1"] = compute_util_metric("FPU_INSTRN_ISSUED_1", scale=1)
    if has_counter("PACK_INSTRN_ISSUED_2"):
        per_op_stats["Pack Instrn Issue Rate T2"] = compute_util_metric("PACK_INSTRN_ISSUED_2", scale=1)

    # === L1 Bank 1 ===
    if has_counter("L1_1_NOC_RING1_OUTGOING_0") and has_counter("L1_1_NOC_RING1_OUTGOING_1"):
        per_op_stats["NOC Ring 1 Outgoing Util"] = compute_avg_channel_util(
            "L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1"
        )
    if has_counter("L1_1_NOC_RING1_INCOMING_0") and has_counter("L1_1_NOC_RING1_INCOMING_1"):
        per_op_stats["NOC Ring 1 Incoming Util"] = compute_avg_channel_util(
            "L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1"
        )

    # === Derived stall metrics (req - grant) / req * 100 ===
    if has_counter("L1_1_NOC_RING1_OUTGOING_0") and has_counter("L1_1_NOC_RING1_OUTGOING_0_GRANT"):
        per_op_stats["NOC Ring 1 Outgoing Backpressure"] = compute_backpressure(
            "L1_1_NOC_RING1_OUTGOING_0",
            "L1_1_NOC_RING1_OUTGOING_1",
            "L1_1_NOC_RING1_OUTGOING_0_GRANT",
            "L1_1_NOC_RING1_OUTGOING_1_GRANT",
        )
    if has_counter("L1_1_NOC_RING1_INCOMING_0") and has_counter("L1_1_NOC_RING1_INCOMING_0_GRANT"):
        per_op_stats["NOC Ring 1 Incoming Backpressure"] = compute_backpressure(
            "L1_1_NOC_RING1_INCOMING_0",
            "L1_1_NOC_RING1_INCOMING_1",
            "L1_1_NOC_RING1_INCOMING_0_GRANT",
            "L1_1_NOC_RING1_INCOMING_1_GRANT",
        )
    if has_counter("L1_0_UNPACKER_0") and has_counter("L1_0_UNPACKER_0_GRANT"):
        req = get_counter_series("L1_0_UNPACKER_0")
        grant = get_counter_series("L1_0_UNPACKER_0_GRANT")
        # On BH, the L1 unpacker grant counter measures a different event than
        # the req counter (grant ~25-45 while req ~8000+, and grant > req on some
        # cores). The resulting 96-100% backpressure is an artifact of signal
        # mismatch, not real L1 contention. Only compute when the median
        # grant/req ratio is reasonable (>10%), indicating the counters track
        # related events.
        valid = req[req > 0]
        grant_valid = grant[req > 0]
        median_ratio = (grant_valid / valid).median() if len(valid) > 0 else 0
        if median_ratio > 0.1:
            ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
            per_op_stats["L1 Unpacker Backpressure"] = _group_to_stat_dict(ratio)
    if has_counter("L1_0_UNIFIED_PACKER") and has_counter("L1_0_PORT1_GRANT"):
        req = get_counter_series("L1_0_UNIFIED_PACKER")
        grant = get_counter_series("L1_0_PORT1_GRANT")
        ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
        per_op_stats["L1 Packer Port Backpressure"] = _group_to_stat_dict(ratio)
    elif has_counter("L1_0_UNPACKER_1_ECC_PACK1") and has_counter("L1_0_PORT1_GRANT"):
        req = get_counter_series("L1_0_UNPACKER_1_ECC_PACK1")
        grant = get_counter_series("L1_0_PORT1_GRANT")
        ratio = ((req - grant) / req * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
        per_op_stats["L1 Packer Port Backpressure"] = _group_to_stat_dict(ratio)

    # === Per-type instruction issue efficiency ===
    if has_counter("CFG_INSTRN_AVAILABLE_0"):
        per_op_stats["CFG Instrn Avail Rate T0"] = compute_util_metric("CFG_INSTRN_AVAILABLE_0")
    if has_counter("SYNC_INSTRN_AVAILABLE_0"):
        per_op_stats["SYNC Instrn Avail Rate T0"] = compute_util_metric("SYNC_INSTRN_AVAILABLE_0")
    if has_counter("THCON_INSTRN_AVAILABLE_0"):
        per_op_stats["THCON Instrn Avail Rate T0"] = compute_util_metric("THCON_INSTRN_AVAILABLE_0")
    if has_counter("MOVE_INSTRN_AVAILABLE_0"):
        per_op_stats["MOVE Instrn Avail Rate T0"] = compute_util_metric("MOVE_INSTRN_AVAILABLE_0")
    if has_counter("FPU_INSTRN_AVAILABLE_1"):
        per_op_stats["MATH Instrn Avail Rate T1"] = compute_util_metric("FPU_INSTRN_AVAILABLE_1")
    if has_counter("UNPACK_INSTRN_AVAILABLE_0"):
        per_op_stats["UNPACK Instrn Avail Rate T0"] = compute_util_metric("UNPACK_INSTRN_AVAILABLE_0")
    if has_counter("PACK_INSTRN_AVAILABLE_2"):
        per_op_stats["PACK Instrn Avail Rate T2"] = compute_util_metric("PACK_INSTRN_AVAILABLE_2")

    # === Write port blocking ===
    if has_counter("SRCB_WRITE_AVAILABLE") and has_counter("SRCB_WRITE_NOT_BLOCKED_PORT"):
        per_op_stats["SrcB Write Port Blocked Rate"] = compute_complement_metric(
            "SRCB_WRITE_NOT_BLOCKED_PORT", "SRCB_WRITE_AVAILABLE"
        )
    if has_counter("SRCA_WRITE_ACTUAL") and has_counter("SRCA_WRITE_AVAILABLE"):
        per_op_stats["SrcA Write Actual Efficiency"] = compute_ratio_metric("SRCA_WRITE_ACTUAL", "SRCA_WRITE_AVAILABLE")
    if has_counter("SRCB_WRITE_ACTUAL") and has_counter("SRCB_WRITE_AVAILABLE"):
        per_op_stats["SrcB Write Actual Efficiency"] = compute_ratio_metric("SRCB_WRITE_ACTUAL", "SRCB_WRITE_AVAILABLE")

    # === Fidelity analysis ===
    if has_counter("MATH_INSTRN_STARTED") and has_counter("INSTRN_2_HF_CYCLES") and has_counter("INSTRN_1_HF_CYCLE"):
        total = get_counter_series("MATH_INSTRN_STARTED")
        hf2 = get_counter_series("INSTRN_2_HF_CYCLES")
        hf1 = get_counter_series("INSTRN_1_HF_CYCLE")
        # On WH, MATH_INSTRN_NOT_BLOCKED_SRC (counter_sel 256) is actually the 4-HF-cycle
        # counter (o_math_instrnbuf_rden & hf_cycles==2'b11). Use it directly when available.
        # On BH, counter_sel 256 is dead (o_math_instrnbuf_rden inactive), so derive by subtraction.
        if has_counter("MATH_INSTRN_NOT_BLOCKED_SRC"):
            hf4_direct = get_counter_series("MATH_INSTRN_NOT_BLOCKED_SRC")
            # If the counter has data and total > 0, use it directly (WH).
            # If it's all zeros (BH dead), fall back to derivation.
            if hf4_direct is not None and hf4_direct.sum() > 0:
                hf4 = hf4_direct
            else:
                hf4 = total - hf2 - hf1
        else:
            hf4 = total - hf2 - hf1
        ratio = (hf4 / total * 100).clip(lower=0).replace([float("inf"), -float("inf")], nan)
        per_op_stats["HiFi4 Instrn Rate"] = _group_to_stat_dict(ratio)
    if has_counter("FIDELITY_PHASE_STALLS"):
        per_op_stats["Fidelity Phase Overhead"] = compute_util_metric("FIDELITY_PHASE_STALLS")

    # === Packer engine granularity ===
    if has_counter("PACKER_BUSY_0"):
        per_op_stats["Packer Engine 0 Util"] = compute_util_metric("PACKER_BUSY_0")
    if has_counter("PACKER_BUSY_1"):
        per_op_stats["Packer Engine 1 Util"] = compute_util_metric("PACKER_BUSY_1")
    if has_counter("PACKER_BUSY_2"):
        per_op_stats["Packer Engine 2 Util"] = compute_util_metric("PACKER_BUSY_2")

    # === Low priority waits ===
    if has_counter("WAITING_FOR_MMIO_IDLE_0"):
        per_op_stats["MMIO Idle Wait T0"] = compute_util_metric("WAITING_FOR_MMIO_IDLE_0")
    if has_counter("WAITING_FOR_SFPU_IDLE_1"):
        per_op_stats["SFPU Idle Wait T1"] = compute_util_metric("WAITING_FOR_SFPU_IDLE_1")
    if has_counter("WAITING_FOR_THCON_IDLE_0"):
        per_op_stats["THCON Idle Wait T0"] = compute_util_metric("WAITING_FOR_THCON_IDLE_0")
    if has_counter("WAITING_FOR_MOVE_IDLE_0"):
        per_op_stats["MOVE Idle Wait T0"] = compute_util_metric("WAITING_FOR_MOVE_IDLE_0")
    if has_counter("L1_1_RISC_CORE"):
        per_op_stats["RISC Core L1 Util"] = compute_util_metric("L1_1_RISC_CORE")

    # === L1 composite metrics (multi-counter) ===
    if has_counter("L1_0_UNPACKER_0") and has_counter("L1_0_NOC_RING0_OUTGOING_0"):
        # L1 Total Bandwidth Util: sum of all 8 port reqs / (8 * ref_cnt)
        packer_key = "L1_0_UNIFIED_PACKER" if has_counter("L1_0_UNIFIED_PACKER") else "L1_0_UNPACKER_1_ECC_PACK1"
        port_keys = [
            "L1_0_UNPACKER_0",
            packer_key,
            "L1_0_TDMA_BUNDLE_0_RISC",
            "L1_0_TDMA_BUNDLE_1_TRISC",
            "L1_0_NOC_RING0_OUTGOING_0",
            "L1_0_NOC_RING0_OUTGOING_1",
            "L1_0_NOC_RING0_INCOMING_0",
            "L1_0_NOC_RING0_INCOMING_1",
        ]
        total_req = sum(get_counter_series(k) for k in port_keys if has_counter(k))
        ref = get_counter_ref_cnt("L1_0_UNPACKER_0")
        ratio = (total_req / (8 * ref) * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["L1 Total Bandwidth Util"] = _group_to_stat_dict(ratio)

        # L1 Read vs Write Ratio
        reads = (
            get_counter_series("L1_0_UNPACKER_0")
            + get_counter_series("L1_0_NOC_RING0_OUTGOING_0")
            + get_counter_series("L1_0_NOC_RING0_OUTGOING_1")
        )
        writes = (
            get_counter_series(packer_key)
            + get_counter_series("L1_0_NOC_RING0_INCOMING_0")
            + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
        )
        ratio = (reads / (reads + writes) * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["L1 Read vs Write Ratio"] = _group_to_stat_dict(ratio)

        # NOC Ring 0 Asymmetry
        noc_out = get_counter_series("L1_0_NOC_RING0_OUTGOING_0") + get_counter_series("L1_0_NOC_RING0_OUTGOING_1")
        noc_in = get_counter_series("L1_0_NOC_RING0_INCOMING_0") + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
        ratio = (noc_out / (noc_out + noc_in) * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["NOC Ring 0 Asymmetry"] = _group_to_stat_dict(ratio)

        # TDMA vs NOC L1 Share
        tdma = get_counter_series("L1_0_TDMA_BUNDLE_0_RISC") + get_counter_series("L1_0_TDMA_BUNDLE_1_TRISC")
        ratio = (tdma / (tdma + noc_out + noc_in) * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["TDMA vs NOC L1 Share"] = _group_to_stat_dict(ratio)

    if has_counter("L1_0_UNPACKER_0_GRANT") and has_counter("L1_0_NOC_RING0_OUTGOING_0_GRANT"):
        # L1 Contention Index
        bp_pairs = [
            ("L1_0_UNPACKER_0", "L1_0_UNPACKER_0_GRANT"),
            ("L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_0_GRANT"),
            ("L1_0_NOC_RING0_OUTGOING_1", "L1_0_NOC_RING0_OUTGOING_1_GRANT"),
            ("L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_0_GRANT"),
            ("L1_0_NOC_RING0_INCOMING_1", "L1_0_NOC_RING0_INCOMING_1_GRANT"),
        ]
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
            per_op_stats["L1 Contention Index"] = _group_to_stat_dict(avg_bp)

    if has_counter("L1_0_UNPACKER_0_GRANT") and has_counter("UNPACK0_BUSY_THREAD0"):
        per_op_stats["Unpacker L1 Efficiency"] = compute_ratio_metric("L1_0_UNPACKER_0_GRANT", "UNPACK0_BUSY_THREAD0")

    if has_counter("L1_0_PORT1_GRANT") and has_counter("PACKER_BUSY"):
        # Packer port is shared (other traffic uses it too), so cap at 100%
        grant = get_counter_series("L1_0_PORT1_GRANT")
        busy = get_counter_series("PACKER_BUSY")
        ratio = (grant / busy * 100).clip(upper=100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["Packer L1 Efficiency"] = _group_to_stat_dict(ratio)

    if has_counter("FPU_COUNTER") and has_counter("L1_0_NOC_RING0_OUTGOING_0"):
        noc_total = (
            get_counter_series("L1_0_NOC_RING0_OUTGOING_0")
            + get_counter_series("L1_0_NOC_RING0_OUTGOING_1")
            + get_counter_series("L1_0_NOC_RING0_INCOMING_0")
            + get_counter_series("L1_0_NOC_RING0_INCOMING_1")
        )
        fpu = get_counter_series("FPU_COUNTER")
        ratio = (noc_total / (fpu + noc_total) * 100).replace([float("inf"), -float("inf")], nan)
        per_op_stats["NOC vs Compute Balance"] = _group_to_stat_dict(ratio)

    return {"per_op_stats": per_op_stats, "per_op_counts": per_op_counts}


def compute_device_only_metrics(
    perf_counter_df: pd.DataFrame,
    device_arch: str,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Compute device-only efficiency metrics from perf counter data.

    This creates a pivot table from the raw counter dataframe, derives ~40
    efficiency/utilization metrics via per-row lambda functions, aggregates
    them per-op (min/median/max/avg), and builds the summary rows for
    ``print_efficiency_metrics_summary``.

    Parameters
    ----------
    perf_counter_df : pd.DataFrame
        Raw perf counter data with columns ``run_host_id``, ``trace_id_count``,
        ``core_x``, ``core_y``, ``counter type``, ``value``, ``ref cnt``.
    device_arch : str
        Device architecture string (e.g. ``"wormhole_b0"``, ``"blackhole"``).

    Returns
    -------
    tuple of (agg_metrics, eff_summary_rows)
        agg_metrics : dict mapping metric_name -> {min/median/max/avg dicts}
        eff_summary_rows : list of row dicts suitable for
            ``pd.DataFrame(eff_summary_rows)`` → ``print_efficiency_metrics_summary``
    """

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
    eff_pivot.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in eff_pivot.columns.values]

    # Helper function for safe division
    def safe_div(num, denom):
        return (num / denom * 100) if denom > 0 else nan

    # Calculate per-core efficiency metrics
    eff_pivot["SFPU Util"] = eff_pivot.apply(
        lambda x: (
            (x.get("value_SFPU_COUNTER", 0) / x.get("ref_cnt_SFPU_COUNTER", 1) * 100)
            if x.get("ref_cnt_SFPU_COUNTER", 0) > 0
            else nan
        ),
        axis=1,
    )
    eff_pivot["FPU Util"] = eff_pivot.apply(
        lambda x: (
            (x.get("value_FPU_COUNTER", 0) / x.get("ref_cnt_FPU_COUNTER", 1) * 100)
            if x.get("ref_cnt_FPU_COUNTER", 0) > 0
            else nan
        ),
        axis=1,
    )
    eff_pivot["MATH Util"] = eff_pivot.apply(
        lambda x: (
            (x.get("value_MATH_COUNTER", 0) / x.get("ref_cnt_MATH_COUNTER", 1) * 100)
            if x.get("ref_cnt_MATH_COUNTER", 0) > 0
            else nan
        ),
        axis=1,
    )
    # Unpacker Write Efficiency — WH only (SRCA_WRITE/SRCB_WRITE counters not on BH)
    if "value_SRCA_WRITE" in eff_pivot.columns:
        eff_pivot["Unpacker0 Write Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_SRCA_WRITE", 0), x.get("value_UNPACK0_BUSY_THREAD0", 0)),
            axis=1,
        )
        eff_pivot["Unpacker1 Write Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_SRCB_WRITE", 0), x.get("value_UNPACK1_BUSY_THREAD0", 0)),
            axis=1,
        )
    # Packer Efficiency: On WH, PACKER_DEST_READ_AVAILABLE / PACKER_BUSY.
    # On BH, PACKER_BUSY is always 0. Fallback: DEST_READ_GRANTED_1 / PACKER_DEST_READ_AVAILABLE
    # (grant rate for dest reads — both signals track the same event on BH silicon).
    has_packer_busy = "value_PACKER_BUSY" in eff_pivot.columns and eff_pivot["value_PACKER_BUSY"].sum() > 0
    if has_packer_busy:
        eff_pivot["Packer Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_PACKER_DEST_READ_AVAILABLE", 0), x.get("value_PACKER_BUSY", 0)),
            axis=1,
        )
    elif "value_DEST_READ_GRANTED_1" in eff_pivot.columns:
        eff_pivot["Packer Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_DEST_READ_GRANTED_1", 0), x.get("value_PACKER_DEST_READ_AVAILABLE", 0)),
            axis=1,
        )

    # Math Pipeline Utilization: On BH, MATH_INSTRN_STARTED is always 0.
    # Fall back to FIDELITY_PHASE_STALLS / ref_cnt.
    if "value_MATH_INSTRN_STARTED" in eff_pivot.columns and eff_pivot["value_MATH_INSTRN_STARTED"].sum() > 0:
        eff_pivot["Math Pipeline Utilization"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_MATH_INSTRN_STARTED", 0), x.get("value_MATH_INSTRN_AVAILABLE", 0)),
            axis=1,
        )
    elif "value_FIDELITY_PHASE_STALLS" in eff_pivot.columns:
        eff_pivot["Math Pipeline Utilization"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_FIDELITY_PHASE_STALLS", 0), x.get("ref_cnt_FIDELITY_PHASE_STALLS", 0)),
            axis=1,
        )

    # Math-to-Pack Handoff: On BH, PACKER_BUSY is always 0.
    # Fall back to AVAILABLE_MATH / ref_cnt.
    if has_packer_busy:
        eff_pivot["Math-to-Pack Handoff Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_AVAILABLE_MATH", 0), x.get("value_PACKER_BUSY", 0)),
            axis=1,
        )
    elif "ref_cnt_AVAILABLE_MATH" in eff_pivot.columns:
        eff_pivot["Math-to-Pack Handoff Efficiency"] = eff_pivot.apply(
            lambda x: safe_div(x.get("value_AVAILABLE_MATH", 0), x.get("ref_cnt_AVAILABLE_MATH", 0)),
            axis=1,
        )
    eff_pivot["Unpacker-to-Math Data Flow"] = eff_pivot.apply(
        lambda x: safe_div(
            (x.get("value_SRCA_WRITE_AVAILABLE", 0) + x.get("value_SRCB_WRITE_AVAILABLE", 0)) / 2,
            (x.get("value_UNPACK0_BUSY_THREAD0", 0) + x.get("value_UNPACK1_BUSY_THREAD0", 0)) / 2,
        ),
        axis=1,
    )
    if "Unpacker0 Write Efficiency" in eff_pivot.columns:
        eff_pivot["Unpacker Write Efficiency"] = eff_pivot[
            ["Unpacker0 Write Efficiency", "Unpacker1 Write Efficiency"]
        ].mean(axis=1, skipna=True)
    eff_pivot["FPU Execution Efficiency"] = eff_pivot.apply(
        lambda x: (
            (x.get("value_FPU_COUNTER", 0) / x.get("value_FPU_INSTRN_AVAILABLE_1", 1) * 100)
            if x.get("value_FPU_INSTRN_AVAILABLE_1", 0) > 0
            else nan
        ),
        axis=1,
    )

    # New metrics: Thread stall rates and IPC
    for t in range(3):
        stall_col = f"value_THREAD_STALLS_{t}"
        ref_col = f"ref_cnt_THREAD_STALLS_{t}"
        eff_pivot[f"Thread {t} Stall Rate"] = eff_pivot.apply(
            lambda x, s=stall_col, r=ref_col: safe_div(x.get(s, 0), x.get(r, 0)),
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
            lambda x, v=val_col, r=ref_col: safe_div(x.get(v, 0), x.get(r, 0)),
            axis=1,
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
                lambda x, v=val_col, r=ref_col: safe_div(x.get(v, 0), x.get(r, 0)),
                axis=1,
            )

    # Data Hazard Stall Rate
    eff_pivot["Data Hazard Stall Rate"] = eff_pivot.apply(
        lambda x: safe_div(
            x.get("value_DATA_HAZARD_STALLS_MOVD2A", 0),
            x.get("ref_cnt_DATA_HAZARD_STALLS_MOVD2A", 0),
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
            (x.get("value_L1_0_TDMA_BUNDLE_0_RISC", 0) + x.get("value_L1_0_TDMA_BUNDLE_1_TRISC", 0)) / 2,
            x.get("ref_cnt_L1_0_TDMA_BUNDLE_0_RISC", 0),
        ),
        axis=1,
    )
    eff_pivot["NOC Ring 0 Outgoing Util"] = eff_pivot.apply(
        lambda x: safe_div(
            (x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0) + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0)) / 2,
            x.get("ref_cnt_L1_0_NOC_RING0_OUTGOING_0", 0),
        ),
        axis=1,
    )
    eff_pivot["NOC Ring 0 Incoming Util"] = eff_pivot.apply(
        lambda x: safe_div(
            (x.get("value_L1_0_NOC_RING0_INCOMING_0", 0) + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0)) / 2,
            x.get("ref_cnt_L1_0_NOC_RING0_INCOMING_0", 0),
        ),
        axis=1,
    )
    # L1 Bank 1 metrics
    eff_pivot["NOC Ring 1 Outgoing Util"] = eff_pivot.apply(
        lambda x: safe_div(
            (x.get("value_L1_1_NOC_RING1_OUTGOING_0", 0) + x.get("value_L1_1_NOC_RING1_OUTGOING_1", 0)) / 2,
            x.get("ref_cnt_L1_1_NOC_RING1_OUTGOING_0", 0),
        ),
        axis=1,
    )
    eff_pivot["NOC Ring 1 Incoming Util"] = eff_pivot.apply(
        lambda x: safe_div(
            (x.get("value_L1_1_NOC_RING1_INCOMING_0", 0) + x.get("value_L1_1_NOC_RING1_INCOMING_1", 0)) / 2,
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

    # On BH, L1 unpacker grant counter has different signal semantics
    # (grant ~25-45 while req ~8000+). Only compute when median grant/req > 10%.
    _unp_req = eff_pivot.get("value_L1_0_UNPACKER_0", pd.Series(dtype=float))
    _unp_gnt = eff_pivot.get("value_L1_0_UNPACKER_0_GRANT", pd.Series(dtype=float))
    _valid = _unp_req[_unp_req > 0]
    _gnt_valid = _unp_gnt[_unp_req > 0]
    _median_ratio = (_gnt_valid / _valid).median() if len(_valid) > 0 else 0
    if _median_ratio > 0.1:
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

    # === Per-type instruction issue efficiency ===
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
    eff_pivot["CFG Instrn Avail Rate T0"] = eff_pivot.apply(
        safe_util("value_CFG_INSTRN_AVAILABLE_0", "ref_cnt_CFG_INSTRN_AVAILABLE_0"),
        axis=1,
    )
    eff_pivot["SYNC Instrn Avail Rate T0"] = eff_pivot.apply(
        safe_util("value_SYNC_INSTRN_AVAILABLE_0", "ref_cnt_SYNC_INSTRN_AVAILABLE_0"),
        axis=1,
    )
    eff_pivot["THCON Instrn Avail Rate T0"] = eff_pivot.apply(
        safe_util("value_THCON_INSTRN_AVAILABLE_0", "ref_cnt_THCON_INSTRN_AVAILABLE_0"),
        axis=1,
    )
    eff_pivot["MOVE Instrn Avail Rate T0"] = eff_pivot.apply(
        safe_util("value_MOVE_INSTRN_AVAILABLE_0", "ref_cnt_MOVE_INSTRN_AVAILABLE_0"),
        axis=1,
    )
    eff_pivot["MATH Instrn Avail Rate T1"] = eff_pivot.apply(
        safe_util("value_FPU_INSTRN_AVAILABLE_1", "ref_cnt_FPU_INSTRN_AVAILABLE_1"),
        axis=1,
    )
    eff_pivot["UNPACK Instrn Avail Rate T0"] = eff_pivot.apply(
        safe_util("value_UNPACK_INSTRN_AVAILABLE_0", "ref_cnt_UNPACK_INSTRN_AVAILABLE_0"),
        axis=1,
    )
    eff_pivot["PACK Instrn Avail Rate T2"] = eff_pivot.apply(
        safe_util("value_PACK_INSTRN_AVAILABLE_2", "ref_cnt_PACK_INSTRN_AVAILABLE_2"),
        axis=1,
    )

    # Write port blocking
    eff_pivot["SrcA Write Port Blocked Rate"] = eff_pivot.apply(
        safe_complement("value_SRCA_WRITE_NOT_BLOCKED_OVR", "value_SRCA_WRITE_AVAILABLE"),
        axis=1,
    )
    eff_pivot["SrcB Write Port Blocked Rate"] = eff_pivot.apply(
        safe_complement("value_SRCB_WRITE_NOT_BLOCKED_PORT", "value_SRCB_WRITE_AVAILABLE"),
        axis=1,
    )
    eff_pivot["SrcA Write Actual Efficiency"] = eff_pivot.apply(
        safe_ratio("value_SRCA_WRITE_ACTUAL", "value_SRCA_WRITE_AVAILABLE"),
        axis=1,
    )
    eff_pivot["SrcB Write Actual Efficiency"] = eff_pivot.apply(
        safe_ratio("value_SRCB_WRITE_ACTUAL", "value_SRCB_WRITE_AVAILABLE"),
        axis=1,
    )

    # Dest read and math stall metrics
    def safe_bp_single(req_key, grant_key):
        def fn(x):
            r = x.get(req_key, 0)
            g = x.get(grant_key, 0)
            return max(0.0, (r - g) / r * 100) if r > 0 else nan

        return fn

    # BH uses DEST_READ_GRANTED_1 (268), WH uses DEST_READ_GRANTED_0 (267)
    dest_grant_col = (
        "value_DEST_READ_GRANTED_1" if "value_DEST_READ_GRANTED_1" in eff_pivot.columns else "value_DEST_READ_GRANTED_0"
    )
    eff_pivot["Dest Read Backpressure"] = eff_pivot.apply(
        safe_bp_single("value_PACKER_DEST_READ_AVAILABLE", dest_grant_col),
        axis=1,
    )
    if (
        "value_MATH_NOT_STALLED_DEST_WR_PORT" in eff_pivot.columns
        and eff_pivot["value_MATH_NOT_STALLED_DEST_WR_PORT"].sum() > 0
    ):
        eff_pivot["Math Dest Write Port Stall Rate"] = eff_pivot.apply(
            safe_complement("value_MATH_NOT_STALLED_DEST_WR_PORT", "value_MATH_INSTRN_AVAILABLE"),
            axis=1,
        )
    eff_pivot["Math Scoreboard Stall Rate"] = eff_pivot.apply(
        safe_complement("value_AVAILABLE_MATH", "value_MATH_INSTRN_AVAILABLE"),
        axis=1,
    )

    # Instruction issue rates (per cycle, not %)
    eff_pivot["Unpack Instrn Issue Rate T0"] = eff_pivot.apply(
        lambda x: (
            x.get("value_UNPACK_INSTRN_ISSUED_0", 0) / x.get("ref_cnt_UNPACK_INSTRN_ISSUED_0", 1)
            if x.get("ref_cnt_UNPACK_INSTRN_ISSUED_0", 0) > 0
            else nan
        ),
        axis=1,
    )
    eff_pivot["Math Instrn Issue Rate T1"] = eff_pivot.apply(
        lambda x: (
            x.get("value_FPU_INSTRN_ISSUED_1", 0) / x.get("ref_cnt_FPU_INSTRN_ISSUED_1", 1)
            if x.get("ref_cnt_FPU_INSTRN_ISSUED_1", 0) > 0
            else nan
        ),
        axis=1,
    )
    eff_pivot["Pack Instrn Issue Rate T2"] = eff_pivot.apply(
        lambda x: (
            x.get("value_PACK_INSTRN_ISSUED_2", 0) / x.get("ref_cnt_PACK_INSTRN_ISSUED_2", 1)
            if x.get("ref_cnt_PACK_INSTRN_ISSUED_2", 0) > 0
            else nan
        ),
        axis=1,
    )

    # Fidelity analysis
    def hifi4_rate_fn(x):
        total = x.get("value_MATH_INSTRN_STARTED", 0)
        if total <= 0:
            return nan
        # On WH, MATH_INSTRN_NOT_BLOCKED_SRC (counter_sel 256) is the 4-HF-cycle counter
        hf4_direct = x.get("value_MATH_INSTRN_NOT_BLOCKED_SRC", 0)
        if hf4_direct > 0:
            return max(0.0, hf4_direct / total * 100)
        hf2 = x.get("value_INSTRN_2_HF_CYCLES", 0)
        hf1 = x.get("value_INSTRN_1_HF_CYCLE", 0)
        return max(0.0, (total - hf2 - hf1) / total * 100)

    eff_pivot["HiFi4 Instrn Rate"] = eff_pivot.apply(hifi4_rate_fn, axis=1)
    eff_pivot["Fidelity Phase Overhead"] = eff_pivot.apply(
        safe_util("value_FIDELITY_PHASE_STALLS", "ref_cnt_FIDELITY_PHASE_STALLS"),
        axis=1,
    )

    # Packer engine granularity (WH only — BH has PACK_COUNT=1, counters not collected)
    if "value_PACKER_BUSY_0" in eff_pivot.columns:
        eff_pivot["Packer Engine 0 Util"] = eff_pivot.apply(
            safe_util("value_PACKER_BUSY_0", "ref_cnt_PACKER_BUSY_0"), axis=1
        )
        eff_pivot["Packer Engine 1 Util"] = eff_pivot.apply(
            safe_util("value_PACKER_BUSY_1", "ref_cnt_PACKER_BUSY_1"), axis=1
        )
        eff_pivot["Packer Engine 2 Util"] = eff_pivot.apply(
            safe_util("value_PACKER_BUSY_2", "ref_cnt_PACKER_BUSY_2"), axis=1
        )

    # Low priority waits
    eff_pivot["MMIO Idle Wait T0"] = eff_pivot.apply(
        safe_util("value_WAITING_FOR_MMIO_IDLE_0", "ref_cnt_WAITING_FOR_MMIO_IDLE_0"),
        axis=1,
    )
    eff_pivot["SFPU Idle Wait T1"] = eff_pivot.apply(
        safe_util("value_WAITING_FOR_SFPU_IDLE_1", "ref_cnt_WAITING_FOR_SFPU_IDLE_1"),
        axis=1,
    )
    eff_pivot["THCON Idle Wait T0"] = eff_pivot.apply(
        safe_util("value_WAITING_FOR_THCON_IDLE_0", "ref_cnt_WAITING_FOR_THCON_IDLE_0"),
        axis=1,
    )
    eff_pivot["MOVE Idle Wait T0"] = eff_pivot.apply(
        safe_util("value_WAITING_FOR_MOVE_IDLE_0", "ref_cnt_WAITING_FOR_MOVE_IDLE_0"),
        axis=1,
    )
    eff_pivot["RISC Core L1 Util"] = eff_pivot.apply(
        safe_util("value_L1_1_RISC_CORE", "ref_cnt_L1_1_RISC_CORE"), axis=1
    )

    # === L1 composite metrics (multi-counter) ===
    def l1_total_bw(x):
        """Sum of all 8 L1_0 port req counts / (8 * ref_cnt)."""
        ports = [
            "value_L1_0_UNPACKER_0",
            (
                "value_L1_0_UNIFIED_PACKER"
                if "value_L1_0_UNIFIED_PACKER" in eff_pivot.columns
                else "value_L1_0_UNPACKER_1_ECC_PACK1"
            ),
            "value_L1_0_TDMA_BUNDLE_0_RISC",
            "value_L1_0_TDMA_BUNDLE_1_TRISC",
            "value_L1_0_NOC_RING0_OUTGOING_0",
            "value_L1_0_NOC_RING0_OUTGOING_1",
            "value_L1_0_NOC_RING0_INCOMING_0",
            "value_L1_0_NOC_RING0_INCOMING_1",
        ]
        total = sum(x.get(p, 0) for p in ports)
        ref = x.get("ref_cnt_L1_0_UNPACKER_0", 0)
        return (total / (8 * ref) * 100) if ref > 0 else nan

    eff_pivot["L1 Total Bandwidth Util"] = eff_pivot.apply(l1_total_bw, axis=1)

    def l1_rw_ratio(x):
        """(read ports) / (write ports). Read = Unpacker + NOC Out, Write = Packer + NOC In."""
        reads = (
            x.get("value_L1_0_UNPACKER_0", 0)
            + x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0)
            + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0)
        )
        writes = (
            x.get("value_L1_0_UNIFIED_PACKER", x.get("value_L1_0_UNPACKER_1_ECC_PACK1", 0))
            + x.get("value_L1_0_NOC_RING0_INCOMING_0", 0)
            + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0)
        )
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
        noc = (
            x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0)
            + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0)
            + x.get("value_L1_0_NOC_RING0_INCOMING_0", 0)
            + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0)
        )
        fpu = x.get("value_FPU_COUNTER", 0)
        total = fpu + noc
        return (noc / total * 100) if total > 0 else nan

    eff_pivot["NOC vs Compute Balance"] = eff_pivot.apply(noc_vs_compute, axis=1)

    def tdma_vs_noc_share(x):
        """TDMA L1 share = TDMA / (TDMA + NOC). Shows RISC vs NOC memory traffic split."""
        tdma = x.get("value_L1_0_TDMA_BUNDLE_0_RISC", 0) + x.get("value_L1_0_TDMA_BUNDLE_1_TRISC", 0)
        noc = (
            x.get("value_L1_0_NOC_RING0_OUTGOING_0", 0)
            + x.get("value_L1_0_NOC_RING0_OUTGOING_1", 0)
            + x.get("value_L1_0_NOC_RING0_INCOMING_0", 0)
            + x.get("value_L1_0_NOC_RING0_INCOMING_1", 0)
        )
        total = tdma + noc
        return (tdma / total * 100) if total > 0 else nan

    eff_pivot["TDMA vs NOC L1 Share"] = eff_pivot.apply(tdma_vs_noc_share, axis=1)

    # Aggregate metrics per operation (min, median, max, avg)
    grouped_eff = eff_pivot.groupby(["run_host_id", "trace_id_count"])

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
        "Unpack Instrn Issue Rate T0",
        "Math Instrn Issue Rate T1",
        "Pack Instrn Issue Rate T2",
    ]

    agg_metrics: Dict[str, Dict] = {}
    for base_name in _pct_metric_names + _ipc_metric_names:
        if base_name in eff_pivot.columns:
            agg_metrics[base_name] = {
                "min": grouped_eff[base_name].min().to_dict(),
                "median": grouped_eff[base_name].median().to_dict(),
                "max": grouped_eff[base_name].max().to_dict(),
                "avg": grouped_eff[base_name].mean().to_dict(),
            }

    # Build efficiency summary rows
    eff_summary_rows: List[Dict] = []
    first_metric = next(iter(agg_metrics.values()), {})
    first_stat = first_metric.get("min", {})
    for key in first_stat.keys():
        row: Dict[str, object] = {}
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
        eff_summary_rows.append(row)

    return agg_metrics, eff_summary_rows


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
