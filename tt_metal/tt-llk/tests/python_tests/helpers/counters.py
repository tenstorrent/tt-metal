# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pandas as pd
from loguru import logger
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .test_config import TestConfig

# Constants and Configuration (derived from TestConfig).

COUNTER_SLOT_COUNT = TestConfig._PERF_COUNTERS_CONFIG_WORDS
COUNTER_DATA_WORD_COUNT = TestConfig._PERF_COUNTERS_DATA_WORDS

PERF_COUNTERS_CONFIG_ADDR = TestConfig.PERF_COUNTERS_CONFIG_ADDR
PERF_COUNTERS_DATA_ADDR = TestConfig.PERF_COUNTERS_DATA_ADDR
PERF_COUNTERS_SYNC_CTRL_ADDR = TestConfig.PERF_COUNTERS_SYNC_CTRL_ADDR
PERF_COUNTERS_START_COUNTER_ADDR = TestConfig.PERF_COUNTERS_SYNC_CTRL_ADDR + 4

PERF_COUNTERS_THREAD_COUNT = len(TestConfig.KERNEL_COMPONENTS)
PERF_COUNTERS_STOP_COUNTER_ADDR = PERF_COUNTERS_START_COUNTER_ADDR + (
    PERF_COUNTERS_THREAD_COUNT * 4
)
PERF_COUNTERS_STOP_ELECT_ADDR = PERF_COUNTERS_STOP_COUNTER_ADDR + (
    PERF_COUNTERS_THREAD_COUNT * 4
)

COUNTER_BANK_NAMES = {
    0: "INSTRN_THREAD",
    1: "FPU",
    2: "TDMA_UNPACK",
    3: "L1",
    4: "TDMA_PACK",
}

# Reverse lookup: bank name -> bank id (computed once at module load)
_BANK_NAME_TO_ID = {v: k for k, v in COUNTER_BANK_NAMES.items()}

# Config word layout (must match counters.h:
# PERF_CFG_VALID_BIT / PERF_CFG_L1_MUX_SHIFT / PERF_CFG_COUNTER_SHIFT / PERF_CFG_BANK_MASK).
PERF_CFG_VALID_BIT = 1 << 31
PERF_CFG_L1_MUX_SHIFT = 17
PERF_CFG_L1_MUX_MASK = 0x7
PERF_CFG_COUNTER_SHIFT = 8
PERF_CFG_COUNTER_MASK = 0x1FF
PERF_CFG_BANK_MASK = 0xFF

# Per-arch counter inventories. WH/BH differ in TDMA_PACK count, L1 mux width, INSTRN_THREAD wait layout.

# Banks shared between WH and BH (identical IDs).
_FPU_COUNTERS = {
    0: "FPU_COUNTER",
    1: "SFPU_COUNTER",
    257: "MATH_COUNTER",
}

_TDMA_UNPACK_COUNTERS = {
    0: "MATH_SRC_DATA_READY",
    1: "DATA_HAZARD_STALLS_MOVD2A",
    2: "MATH_FIDELITY_STALL",
    3: "MATH_INSTRN_STARTED",
    4: "MATH_INSTRN_AVAILABLE",
    5: "SRCB_WRITE_AVAILABLE",
    6: "SRCA_WRITE_AVAILABLE",
    7: "UNPACK0_BUSY_THREAD0",
    8: "UNPACK1_BUSY_THREAD0",
    9: "UNPACK0_BUSY_THREAD1",
    10: "UNPACK1_BUSY_THREAD1",
    256: "MATH_INSTRN_HF_4_CYCLE",
    257: "MATH_INSTRN_HF_2_CYCLE",
    258: "MATH_INSTRN_HF_1_CYCLE",
    259: "SRCB_WRITE_ACTUAL",
    260: "SRCB_WRITE_NOT_BLOCKED_PORT",
    261: "SRCA_WRITE_NOT_BLOCKED_OVR",
    262: "SRCA_WRITE_ACTUAL",
    263: "SRCA_WRITE_THREAD0",
    264: "SRCB_WRITE_THREAD0",
    265: "SRCA_WRITE_THREAD1",
    266: "SRCB_WRITE_THREAD1",
}

# Wormhole-specific tables ===================================================
# Source: tt_metal/hw/inc/internal/tt-1xx/wormhole/hw_counters.h

_WH_TDMA_PACK_COUNTERS = {
    11: "PACKER_DEST_READ_AVAILABLE",
    12: "PACKER_DEST_READ_1",
    13: "PACKER_DEST_READ_2",
    14: "PACKER_DEST_READ_3",
    15: "PACKER_BUSY_0",
    16: "PACKER_BUSY_1",
    17: "PACKER_BUSY_2",
    18: "PACKER_BUSY",
    267: "DEST_READ_GRANTED_0",
    268: "DEST_READ_GRANTED_1",
    269: "DEST_READ_GRANTED_2",
    270: "DEST_READ_GRANTED_3",
    271: "MATH_NOT_STALLED_DEST_WR_PORT",
    272: "AVAILABLE_MATH",
}

# WH INSTRN_THREAD: gap IDs at 9-11 and replicated stall conditions at 27/30/33/36.
_WH_INSTRN_COUNTERS = {
    0: "CFG_INSTRN_AVAILABLE_0",
    1: "CFG_INSTRN_AVAILABLE_1",
    2: "CFG_INSTRN_AVAILABLE_2",
    3: "SYNC_INSTRN_AVAILABLE_0",
    4: "SYNC_INSTRN_AVAILABLE_1",
    5: "SYNC_INSTRN_AVAILABLE_2",
    6: "THCON_INSTRN_AVAILABLE_0",
    7: "THCON_INSTRN_AVAILABLE_1",
    8: "THCON_INSTRN_AVAILABLE_2",
    12: "MOVE_INSTRN_AVAILABLE_0",
    13: "MOVE_INSTRN_AVAILABLE_1",
    14: "MOVE_INSTRN_AVAILABLE_2",
    15: "FPU_INSTRN_AVAILABLE_0",
    16: "FPU_INSTRN_AVAILABLE_1",
    17: "FPU_INSTRN_AVAILABLE_2",
    18: "UNPACK_INSTRN_AVAILABLE_0",
    19: "UNPACK_INSTRN_AVAILABLE_1",
    20: "UNPACK_INSTRN_AVAILABLE_2",
    21: "PACK_INSTRN_AVAILABLE_0",
    22: "PACK_INSTRN_AVAILABLE_1",
    23: "PACK_INSTRN_AVAILABLE_2",
    24: "THREAD_STALLS_0",
    25: "THREAD_STALLS_1",
    26: "THREAD_STALLS_2",
    27: "WAITING_FOR_SRCA_CLEAR",
    30: "WAITING_FOR_SRCB_CLEAR",
    33: "WAITING_FOR_SRCA_VALID",
    36: "WAITING_FOR_SRCB_VALID",
    39: "WAITING_FOR_THCON_IDLE_0",
    40: "WAITING_FOR_UNPACK_IDLE_0",
    41: "WAITING_FOR_PACK_IDLE_0",
    42: "WAITING_FOR_MATH_IDLE_0",
    43: "WAITING_FOR_NONZERO_SEM_0",
    44: "WAITING_FOR_NONFULL_SEM_0",
    45: "WAITING_FOR_MOVE_IDLE_0",
    46: "WAITING_FOR_MMIO_IDLE_0",
    47: "WAITING_FOR_SFPU_IDLE_0",
    48: "WAITING_FOR_THCON_IDLE_1",
    49: "WAITING_FOR_UNPACK_IDLE_1",
    50: "WAITING_FOR_PACK_IDLE_1",
    51: "WAITING_FOR_MATH_IDLE_1",
    52: "WAITING_FOR_NONZERO_SEM_1",
    53: "WAITING_FOR_NONFULL_SEM_1",
    54: "WAITING_FOR_MOVE_IDLE_1",
    55: "WAITING_FOR_MMIO_IDLE_1",
    56: "WAITING_FOR_SFPU_IDLE_1",
    57: "WAITING_FOR_THCON_IDLE_2",
    58: "WAITING_FOR_UNPACK_IDLE_2",
    59: "WAITING_FOR_PACK_IDLE_2",
    60: "WAITING_FOR_MATH_IDLE_2",
    61: "WAITING_FOR_NONZERO_SEM_2",
    62: "WAITING_FOR_NONFULL_SEM_2",
    63: "WAITING_FOR_MOVE_IDLE_2",
    64: "WAITING_FOR_MMIO_IDLE_2",
    65: "WAITING_FOR_SFPU_IDLE_2",
    256: "THREAD_INSTRUCTIONS_0",
    264: "THREAD_INSTRUCTIONS_1",
    272: "THREAD_INSTRUCTIONS_2",
    283: "ANY_THREAD_STALL",
}

# WH L1: 2 banks (1-bit mux). bank 0 = port 1 ECC/Pack1, bank 1 = TDMA Packer 2.
_WH_L1_COUNTERS = {
    (0, 0): "L1_0_UNPACKER_0",
    (1, 0): "L1_0_UNPACKER_1_ECC_PACK1",
    (2, 0): "L1_0_TDMA_BUNDLE_0_RISC",
    (3, 0): "L1_0_TDMA_BUNDLE_1_TRISC",
    (4, 0): "L1_0_NOC_RING0_OUTGOING_0",
    (5, 0): "L1_0_NOC_RING0_OUTGOING_1",
    (6, 0): "L1_0_NOC_RING0_INCOMING_0",
    (7, 0): "L1_0_NOC_RING0_INCOMING_1",
    (256, 0): "L1_0_UNPACKER_0_GRANT",
    (257, 0): "L1_0_PORT1_GRANT",
    (258, 0): "L1_0_TDMA_BUNDLE_0_GRANT",
    (259, 0): "L1_0_TDMA_BUNDLE_1_GRANT",
    (260, 0): "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    (261, 0): "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    (262, 0): "L1_0_NOC_RING0_INCOMING_0_GRANT",
    (263, 0): "L1_0_NOC_RING0_INCOMING_1_GRANT",
    (0, 1): "L1_1_TDMA_PACKER_2",
    (1, 1): "L1_1_EXT_UNPACKER_1",
    (2, 1): "L1_1_EXT_UNPACKER_2",
    (3, 1): "L1_1_EXT_UNPACKER_3",
    (4, 1): "L1_1_NOC_RING1_OUTGOING_0",
    (5, 1): "L1_1_NOC_RING1_OUTGOING_1",
    (6, 1): "L1_1_NOC_RING1_INCOMING_0",
    (7, 1): "L1_1_NOC_RING1_INCOMING_1",
    (256, 1): "L1_1_PORT8_GRANT",
    (257, 1): "L1_1_EXT_UNPACKER_1_GRANT",
    (258, 1): "L1_1_EXT_UNPACKER_2_GRANT",
    (259, 1): "L1_1_EXT_UNPACKER_3_GRANT",
    (260, 1): "L1_1_NOC_RING1_OUTGOING_0_GRANT",
    (261, 1): "L1_1_NOC_RING1_OUTGOING_1_GRANT",
    (262, 1): "L1_1_NOC_RING1_INCOMING_0_GRANT",
    (263, 1): "L1_1_NOC_RING1_INCOMING_1_GRANT",
}

_WORMHOLE_COUNTER_NAMES = {
    "INSTRN_THREAD": _WH_INSTRN_COUNTERS,
    "FPU": _FPU_COUNTERS,
    "TDMA_UNPACK": _TDMA_UNPACK_COUNTERS,
    "TDMA_PACK": _WH_TDMA_PACK_COUNTERS,
    "L1": _WH_L1_COUNTERS,
}

# Blackhole-specific tables ==================================================
# Source: tt_metal/hw/inc/internal/tt-1xx/blackhole/hw_counters.h

# BH: only 1 packer engine.
_BH_TDMA_PACK_COUNTERS = {
    11: "PACKER_DEST_READ_AVAILABLE",
    18: "PACKER_BUSY",
    267: "DEST_READ_GRANTED_0",
    271: "MATH_NOT_STALLED_DEST_WR_PORT",
    272: "AVAILABLE_MATH",
}

# BH INSTRN_THREAD: gap IDs at 9-11 only (kick tied to 0), then contiguous 27..57.
_BH_INSTRN_COUNTERS = {
    0: "CFG_INSTRN_AVAILABLE_0",
    1: "CFG_INSTRN_AVAILABLE_1",
    2: "CFG_INSTRN_AVAILABLE_2",
    3: "SYNC_INSTRN_AVAILABLE_0",
    4: "SYNC_INSTRN_AVAILABLE_1",
    5: "SYNC_INSTRN_AVAILABLE_2",
    6: "THCON_INSTRN_AVAILABLE_0",
    7: "THCON_INSTRN_AVAILABLE_1",
    8: "THCON_INSTRN_AVAILABLE_2",
    12: "MOVE_INSTRN_AVAILABLE_0",
    13: "MOVE_INSTRN_AVAILABLE_1",
    14: "MOVE_INSTRN_AVAILABLE_2",
    15: "FPU_INSTRN_AVAILABLE_0",
    16: "FPU_INSTRN_AVAILABLE_1",
    17: "FPU_INSTRN_AVAILABLE_2",
    18: "UNPACK_INSTRN_AVAILABLE_0",
    19: "UNPACK_INSTRN_AVAILABLE_1",
    20: "UNPACK_INSTRN_AVAILABLE_2",
    21: "PACK_INSTRN_AVAILABLE_0",
    22: "PACK_INSTRN_AVAILABLE_1",
    23: "PACK_INSTRN_AVAILABLE_2",
    24: "THREAD_STALLS_0",
    25: "THREAD_STALLS_1",
    26: "THREAD_STALLS_2",
    27: "WAITING_FOR_SRCA_CLEAR",
    28: "WAITING_FOR_SRCB_CLEAR",
    29: "WAITING_FOR_SRCA_VALID",
    30: "WAITING_FOR_SRCB_VALID",
    31: "WAITING_FOR_THCON_IDLE_0",
    32: "WAITING_FOR_UNPACK_IDLE_0",
    33: "WAITING_FOR_PACK_IDLE_0",
    34: "WAITING_FOR_MATH_IDLE_0",
    35: "WAITING_FOR_NONZERO_SEM_0",
    36: "WAITING_FOR_NONFULL_SEM_0",
    37: "WAITING_FOR_MOVE_IDLE_0",
    38: "WAITING_FOR_MMIO_IDLE_0",
    39: "WAITING_FOR_SFPU_IDLE_0",
    40: "WAITING_FOR_THCON_IDLE_1",
    41: "WAITING_FOR_UNPACK_IDLE_1",
    42: "WAITING_FOR_PACK_IDLE_1",
    43: "WAITING_FOR_MATH_IDLE_1",
    44: "WAITING_FOR_NONZERO_SEM_1",
    45: "WAITING_FOR_NONFULL_SEM_1",
    46: "WAITING_FOR_MOVE_IDLE_1",
    47: "WAITING_FOR_MMIO_IDLE_1",
    48: "WAITING_FOR_SFPU_IDLE_1",
    49: "WAITING_FOR_THCON_IDLE_2",
    50: "WAITING_FOR_UNPACK_IDLE_2",
    51: "WAITING_FOR_PACK_IDLE_2",
    52: "WAITING_FOR_MATH_IDLE_2",
    53: "WAITING_FOR_NONZERO_SEM_2",
    54: "WAITING_FOR_NONFULL_SEM_2",
    55: "WAITING_FOR_MOVE_IDLE_2",
    56: "WAITING_FOR_MMIO_IDLE_2",
    57: "WAITING_FOR_SFPU_IDLE_2",
    256: "THREAD_INSTRUCTIONS_0",
    264: "THREAD_INSTRUCTIONS_1",
    272: "THREAD_INSTRUCTIONS_2",
    283: "ANY_THREAD_STALL",
}

# BH L1: 5 banks (3-bit mux 0..4). bank 0 = unified packer, bank 1 = RISC core,
# banks 2/3 = NOC Ring 2/3 ports, bank 4 = misc ports.
_BH_L1_COUNTERS = {
    # Bank 0 (mux=0): unpacker, TDMA bundles, NOC Ring 0
    (0, 0): "L1_0_UNPACKER_0",
    (1, 0): "L1_0_UNIFIED_PACKER",
    (2, 0): "L1_0_TDMA_BUNDLE_0_RISC",
    (3, 0): "L1_0_TDMA_BUNDLE_1_TRISC",
    (4, 0): "L1_0_NOC_RING0_OUTGOING_0",
    (5, 0): "L1_0_NOC_RING0_OUTGOING_1",
    (6, 0): "L1_0_NOC_RING0_INCOMING_0",
    (7, 0): "L1_0_NOC_RING0_INCOMING_1",
    (256, 0): "L1_0_UNPACKER_0_GRANT",
    (257, 0): "L1_0_PORT1_GRANT",
    (258, 0): "L1_0_TDMA_BUNDLE_0_GRANT",
    (259, 0): "L1_0_TDMA_BUNDLE_1_GRANT",
    (260, 0): "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    (261, 0): "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    (262, 0): "L1_0_NOC_RING0_INCOMING_0_GRANT",
    (263, 0): "L1_0_NOC_RING0_INCOMING_1_GRANT",
    # Bank 1 (mux=1): RISC core, ext unpacker, NOC Ring 1
    (0, 1): "L1_1_RISC_CORE",
    (1, 1): "L1_1_EXT_UNPACKER_1",
    (2, 1): "L1_1_EXT_UNPACKER_2",
    (3, 1): "L1_1_EXT_UNPACKER_3",
    (4, 1): "L1_1_NOC_RING1_OUTGOING_0",
    (5, 1): "L1_1_NOC_RING1_OUTGOING_1",
    (6, 1): "L1_1_NOC_RING1_INCOMING_0",
    (7, 1): "L1_1_NOC_RING1_INCOMING_1",
    (256, 1): "L1_1_PORT8_GRANT",
    (257, 1): "L1_1_EXT_UNPACKER_1_GRANT",
    (258, 1): "L1_1_EXT_UNPACKER_2_GRANT",
    (259, 1): "L1_1_EXT_UNPACKER_3_GRANT",
    (260, 1): "L1_1_NOC_RING1_OUTGOING_0_GRANT",
    (261, 1): "L1_1_NOC_RING1_OUTGOING_1_GRANT",
    (262, 1): "L1_1_NOC_RING1_INCOMING_0_GRANT",
    (263, 1): "L1_1_NOC_RING1_INCOMING_1_GRANT",
    # Bank 2 (mux=2): NOC Ring 2 ports 16-23 (BH-only)
    (0, 2): "L1_2_NOC_RING2_PORT_0",
    (1, 2): "L1_2_NOC_RING2_PORT_1",
    (2, 2): "L1_2_NOC_RING2_PORT_2",
    (3, 2): "L1_2_NOC_RING2_PORT_3",
    (4, 2): "L1_2_NOC_RING2_PORT_4",
    (5, 2): "L1_2_NOC_RING2_PORT_5",
    (6, 2): "L1_2_NOC_RING2_PORT_6",
    (7, 2): "L1_2_NOC_RING2_PORT_7",
    (256, 2): "L1_2_NOC_RING2_PORT_0_GRANT",
    (257, 2): "L1_2_NOC_RING2_PORT_1_GRANT",
    (258, 2): "L1_2_NOC_RING2_PORT_2_GRANT",
    (259, 2): "L1_2_NOC_RING2_PORT_3_GRANT",
    (260, 2): "L1_2_NOC_RING2_PORT_4_GRANT",
    (261, 2): "L1_2_NOC_RING2_PORT_5_GRANT",
    (262, 2): "L1_2_NOC_RING2_PORT_6_GRANT",
    (263, 2): "L1_2_NOC_RING2_PORT_7_GRANT",
    # Bank 3 (mux=3): NOC Ring 3 ports 24-31 (BH-only)
    (0, 3): "L1_3_NOC_RING3_PORT_0",
    (1, 3): "L1_3_NOC_RING3_PORT_1",
    (2, 3): "L1_3_NOC_RING3_PORT_2",
    (3, 3): "L1_3_NOC_RING3_PORT_3",
    (4, 3): "L1_3_NOC_RING3_PORT_4",
    (5, 3): "L1_3_NOC_RING3_PORT_5",
    (6, 3): "L1_3_NOC_RING3_PORT_6",
    (7, 3): "L1_3_NOC_RING3_PORT_7",
    (256, 3): "L1_3_NOC_RING3_PORT_0_GRANT",
    (257, 3): "L1_3_NOC_RING3_PORT_1_GRANT",
    (258, 3): "L1_3_NOC_RING3_PORT_2_GRANT",
    (259, 3): "L1_3_NOC_RING3_PORT_3_GRANT",
    (260, 3): "L1_3_NOC_RING3_PORT_4_GRANT",
    (261, 3): "L1_3_NOC_RING3_PORT_5_GRANT",
    (262, 3): "L1_3_NOC_RING3_PORT_6_GRANT",
    (263, 3): "L1_3_NOC_RING3_PORT_7_GRANT",
    # Bank 4 (mux=4): misc ports 32-39 (BH-only)
    (0, 4): "L1_4_MISC_PORT_0",
    (1, 4): "L1_4_MISC_PORT_1",
    (2, 4): "L1_4_MISC_PORT_2",
    (3, 4): "L1_4_MISC_PORT_3",
    (4, 4): "L1_4_MISC_PORT_4",
    (5, 4): "L1_4_MISC_PORT_5",
    (6, 4): "L1_4_MISC_PORT_6",
    (7, 4): "L1_4_MISC_PORT_7",
    (256, 4): "L1_4_MISC_PORT_0_GRANT",
    (257, 4): "L1_4_MISC_PORT_1_GRANT",
    (258, 4): "L1_4_MISC_PORT_2_GRANT",
    (259, 4): "L1_4_MISC_PORT_3_GRANT",
    (260, 4): "L1_4_MISC_PORT_4_GRANT",
    (261, 4): "L1_4_MISC_PORT_5_GRANT",
    (262, 4): "L1_4_MISC_PORT_6_GRANT",
    (263, 4): "L1_4_MISC_PORT_7_GRANT",
}

_BLACKHOLE_COUNTER_NAMES = {
    "INSTRN_THREAD": _BH_INSTRN_COUNTERS,
    "FPU": _FPU_COUNTERS,
    "TDMA_UNPACK": _TDMA_UNPACK_COUNTERS,
    "TDMA_PACK": _BH_TDMA_PACK_COUNTERS,
    "L1": _BH_L1_COUNTERS,
}

# Quasar inventory is empty for now — counter layout not yet finalized.
_QUASAR_COUNTER_NAMES = {
    "INSTRN_THREAD": {},
    "FPU": {},
    "TDMA_UNPACK": {},
    "TDMA_PACK": {},
    "L1": {},
}

_arch = get_chip_architecture()
if _arch == ChipArchitecture.WORMHOLE:
    COUNTER_NAMES = _WORMHOLE_COUNTER_NAMES
elif _arch == ChipArchitecture.BLACKHOLE:
    COUNTER_NAMES = _BLACKHOLE_COUNTER_NAMES
else:
    COUNTER_NAMES = _QUASAR_COUNTER_NAMES

# Reverse lookups for O(1) counter name -> id resolution (computed once at module load)
_L1_NAME_TO_ID = {(name, mux): cid for (cid, mux), name in COUNTER_NAMES["L1"].items()}

_COUNTER_NAME_TO_ID = {
    bank: {name: cid for cid, name in counters.items()}
    for bank, counters in COUNTER_NAMES.items()
    if bank != "L1"
}


def _build_all_counters() -> List[Dict]:
    """Build the complete list of performance counters for the active arch.

    L1 entries span multiple mux banks: WH uses mux 0..1, BH mux 0..4. Iterates
    L1 in mux-ascending order so configure_counters writes consecutive mux
    groups together (helps the HW readback path keep the mux setting stable).
    """
    counters = []

    for counter_id in COUNTER_NAMES["INSTRN_THREAD"].keys():
        counters.append({"bank": "INSTRN_THREAD", "counter_id": counter_id})

    for counter_id in COUNTER_NAMES["FPU"].keys():
        counters.append({"bank": "FPU", "counter_id": counter_id})

    for counter_id in COUNTER_NAMES["TDMA_UNPACK"].keys():
        counters.append({"bank": "TDMA_UNPACK", "counter_id": counter_id})

    for counter_id in COUNTER_NAMES["TDMA_PACK"].keys():
        counters.append({"bank": "TDMA_PACK", "counter_id": counter_id})

    # L1: iterate mux groups in ascending order (0, 1, [2, 3, 4 on BH]).
    l1_muxes = sorted({mux for (_, mux) in COUNTER_NAMES["L1"].keys()})
    for mux in l1_muxes:
        for (counter_id, l1_mux), _ in COUNTER_NAMES["L1"].items():
            if l1_mux == mux:
                counters.append({"bank": "L1", "counter_id": counter_id, "l1_mux": mux})

    return counters


# Pre-built list of all counters (WH=130, BH=169).
ALL_COUNTERS = _build_all_counters()


def configure_counters(location: str = "0,0") -> None:
    """Write the per-arch counter inventory into the shared L1 config buffer."""
    config_words = []
    for counter in ALL_COUNTERS:
        l1_mux = counter.get("l1_mux", 0) & PERF_CFG_L1_MUX_MASK
        l1_mux_shifted = l1_mux << PERF_CFG_L1_MUX_SHIFT
        counter_id_shifted = (
            counter["counter_id"] & PERF_CFG_COUNTER_MASK
        ) << PERF_CFG_COUNTER_SHIFT
        bank_id = _BANK_NAME_TO_ID[counter["bank"]] & PERF_CFG_BANK_MASK
        config_words.append(
            PERF_CFG_VALID_BIT | l1_mux_shifted | counter_id_shifted | bank_id
        )

    # Pad config words to full slot count
    config_words.extend([0] * (COUNTER_SLOT_COUNT - len(config_words)))

    # Write config to the single shared buffer (all threads share one location)
    write_words_to_device(
        location=location, addr=PERF_COUNTERS_CONFIG_ADDR, data=config_words
    )

    # Clear data buffer completely (remove any stale values from previous runs)
    write_words_to_device(
        location=location,
        addr=PERF_COUNTERS_DATA_ADDR,
        data=[0] * COUNTER_DATA_WORD_COUNT,
    )

    # Clear sync state and ATINCGET counters before kernel runs (layout matches counters.h).
    # 1 word sync ctrl + PERF_COUNTERS_THREAD_COUNT start + PERF_COUNTERS_THREAD_COUNT stop + 1 word stop_elect
    write_words_to_device(
        location=location,
        addr=PERF_COUNTERS_SYNC_CTRL_ADDR,
        data=[0] * (1 + 2 * PERF_COUNTERS_THREAD_COUNT + 1),
    )


def _zone_config_addr(zone: int) -> int:
    """Config is shared across all zones (single buffer at base addr)."""
    _ = zone
    return TestConfig.PERF_COUNTERS_CONFIG_ADDR


def _zone_data_addr(zone: int) -> int:
    """Per-zone data block start: bank cycles (5 words) then counter counts."""
    return (
        TestConfig.PERF_COUNTERS_ZONES_BASE + zone * TestConfig.PERF_COUNTERS_ZONE_SIZE
    )


def _zone_sync_ctrl_addr(zone: int) -> int:
    """Sync word lives at end of per-zone data block."""
    return _zone_data_addr(zone) + TestConfig._PERF_COUNTERS_ZONE_DATA_BYTES


# Lightweight sync: SYNC_ZONE_COMPLETE marker (matches counters.h)
_SYNC_ZONE_COMPLETE = 0xFF


def _read_zone_counters(location: str, zone: int, zone_name: str) -> list[dict]:
    """
    Read performance counter results for a single zone from L1.

    Returns list of result dicts, or empty list if zone was not used.
    """
    sync_addr = _zone_sync_ctrl_addr(zone)
    sync_ctrl = read_words_from_device(location=location, addr=sync_addr, word_count=1)
    if not sync_ctrl:
        return []

    sync_word = sync_ctrl[0]
    logger.debug(
        f"Zone {zone} ({zone_name}): sync_word=0x{sync_word:08x} at addr=0x{sync_addr:06x}"
    )

    # Zone was never used (BRISC clears sync to 0 before each run)
    if sync_word == 0:
        logger.debug(f"Zone {zone}: sync_word is 0, skipping (zone not used)")
        return []

    # Lightweight stop writes only SYNC_ZONE_COMPLETE (0xFF); the high bytes are
    # unused under the current protocol. Any other low-byte value signals
    # corrupted or partially-written sync state.
    if (sync_word & 0xFF) != _SYNC_ZONE_COMPLETE:
        logger.warning(
            f"Zone {zone}: unexpected sync word 0x{sync_word:08x} "
            f"(expected SYNC_ZONE_COMPLETE=0xFF in low byte)"
        )
        return []

    # Shared config (same for all zones) — read once metadata layout.
    config_addr = _zone_config_addr(zone)
    metadata = read_words_from_device(
        location=location, addr=config_addr, word_count=COUNTER_SLOT_COUNT
    )
    if not metadata:
        return []

    valid_count = sum(1 for m in metadata if (m & PERF_CFG_VALID_BIT) != 0)
    if valid_count == 0:
        return []

    # Per-zone data: 5 bank-cycle words (one OUT_L per bank) + N counter counts.
    data_addr = _zone_data_addr(zone)
    bank_cycles_words = TestConfig._PERF_COUNTERS_BANK_CYCLES_WORDS
    data = read_words_from_device(
        location=location, addr=data_addr, word_count=bank_cycles_words + valid_count
    )
    if len(data) < bank_cycles_words + valid_count:
        return []

    # Bank cycles are the first 5 words: indexed by bank_id (0..4).
    # Order matches counter_bank enum (see counters.h): INSTRN, FPU, TDMA_UNPACK, L1, TDMA_PACK
    bank_cycles = data[:bank_cycles_words]
    counter_counts = data[bank_cycles_words:]

    results = []
    count_idx = 0
    for i in range(COUNTER_SLOT_COUNT):
        config_word = metadata[i]
        if (config_word & PERF_CFG_VALID_BIT) == 0:
            continue

        bank_id = config_word & PERF_CFG_BANK_MASK
        counter_id = (config_word >> PERF_CFG_COUNTER_SHIFT) & PERF_CFG_COUNTER_MASK
        l1_mux = (config_word >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK

        bank_name = COUNTER_BANK_NAMES.get(bank_id, f"UNKNOWN_{bank_id}")

        if bank_name == "L1":
            counter_name = COUNTER_NAMES["L1"].get(
                (counter_id, l1_mux), f"L1_UNKNOWN_{counter_id}_{l1_mux}"
            )
        else:
            counter_name = COUNTER_NAMES.get(bank_name, {}).get(
                counter_id, f"{bank_name}_UNKNOWN_{counter_id}"
            )

        cycles = bank_cycles[bank_id] if bank_id < bank_cycles_words else 0
        count = counter_counts[count_idx]
        count_idx += 1

        results.append(
            {
                "zone": zone_name,
                "bank": bank_name,
                "counter_name": counter_name,
                "counter_id": counter_id,
                "cycles": cycles,
                "count": count,
                "l1_mux": l1_mux if bank_name == "L1" else None,
            }
        )

    return results


def read_counters(location: str = "0,0") -> pd.DataFrame:
    """
    Read performance counter results from all zones.

    Iterates zones 0..MAX_ZONES-1; zones with sync_word=0 (never used) are
    skipped silently. MAX_ZONES is set by counters.h (currently 8).

    Args:
        location: Tensix core coordinates (e.g., "0,0").

    Returns:
        DataFrame with columns:
        zone, bank, counter_name, counter_id, cycles, count, l1_mux
    """
    all_results = []

    for zone_idx in range(TestConfig.PERF_COUNTERS_MAX_ZONES):
        zone_name = f"ZONE_{zone_idx}"
        zone_results = _read_zone_counters(location, zone_idx, zone_name)
        all_results.extend(zone_results)

    return pd.DataFrame(all_results)


def print_counters(results: pd.DataFrame) -> None:
    """
    Log all counter results in a readable format.

    Args:
        results: DataFrame with counter results (from read_counters).
    """
    if results.empty:
        logger.info("No counter results to display.")
        return

    if "zone" in results.columns:
        zones = results["zone"].unique()
        for zone in zones:
            zone_results = results[results["zone"] == zone]
            logger.info("\n{}\nZONE: {}\n{}", "═" * 100, zone, "═" * 100)
            _print_zone_counters(zone_results)
    else:
        logger.info("\n{}\nPERFORMANCE COUNTER RESULTS\n{}", "=" * 100, "=" * 100)
        _print_zone_counters(results)


def _print_zone_counters(results: pd.DataFrame) -> None:
    """Helper to log counters for a single zone."""
    lines = []

    for bank in ["INSTRN_THREAD", "FPU", "TDMA_UNPACK", "L1", "TDMA_PACK"]:
        bank_df = results[results["bank"] == bank]
        if bank_df.empty:
            continue

        cycles = bank_df["cycles"].iloc[0] if len(bank_df) > 0 else 0

        lines.append(f"\n  ┌─ {bank} (cycles: {cycles:,})")
        lines.append(f"  │ {'Counter Name':<40} {'Count':>15} {'Rate':>12}")
        lines.append(f"  │ {'─' * 40} {'─' * 15} {'─' * 12}")

        for _, row in bank_df.iterrows():
            name = row["counter_name"]
            if pd.notna(row["l1_mux"]):
                name = f"{name} (mux{int(row['l1_mux'])})"
            count = row["count"]
            rate = (count / cycles) if cycles else 0.0
            lines.append(f"  │ {name:<40} {count:>15,} {rate:>12.4f}")

        lines.append(f"  └{'─' * 70}")

    if lines:
        logger.info("\n".join(lines))


def export_counters(
    results: pd.DataFrame,
    filename: str,
    test_params: dict = None,
    worker_id: str = "gw0",
) -> None:
    """
    Export counter DataFrame to CSV file in perf_data directory.

    Args:
        results: DataFrame with counter results (from read_counters).
        filename: Base filename (without extension), e.g., "test_matmul_counters".
        test_params: Optional dictionary of test parameters to add as columns.
        worker_id: Worker ID for parallel test runs (e.g., "gw0", "master").
    """
    if results.empty:
        return

    perf_dir = TestConfig.LLK_ROOT / "perf_data"
    perf_dir.mkdir(parents=True, exist_ok=True)

    df = results.copy()

    # Add test params as columns
    if test_params:
        for key, value in test_params.items():
            df[key] = value

    output_path = perf_dir / f"{filename}.{worker_id}.csv"

    if output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(output_path, index=False)
