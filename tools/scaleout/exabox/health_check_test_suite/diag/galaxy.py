# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""BH Galaxy 6U topology: BDF <-> physical slot, ETH port wiring, board revisions.

Constants sourced from tt-system-firmware and tt-smi (see comments).
"""
from __future__ import annotations

import re

# BH Galaxy 6U bus-byte high-nibble -> tray number.
# tt-smi/tt_smi/constants.py:BH_UBB_BUS_IDS = {1: 0x00, 2: 0x40, 3: 0xC0, 4: 0x80}
# Note non-monotonic: 0xC -> 3, 0x8 -> 4.
UBB_NIBBLE_TO_TRAY = {0x0: 1, 0x4: 2, 0xC: 3, 0x8: 4}
TRAY_TO_UBB_NIBBLE = {v: k for k, v in UBB_NIBBLE_TO_TRAY.items()}


def expected_bdf(tray: int, asic_loc: int) -> str:
    """Reconstruct the expected PCIe BDF for a given (tray, asic_loc) physical slot."""
    return f"0000:{TRAY_TO_UBB_NIBBLE[tray]:x}{asic_loc:x}:00.0"


def parse_bdf_physical(bdf: str) -> tuple[int, int] | None:
    """Parse a BDF '0000:XY:00.0' into (tray, asic_loc). None if unparseable."""
    m = re.match(r"^[0-9a-f]+:([0-9a-f])([0-9a-f]):", bdf or "", re.IGNORECASE)
    if not m:
        return None
    tray = UBB_NIBBLE_TO_TRAY.get(int(m.group(1), 16))
    if tray is None:
        return None
    return tray, int(m.group(2), 16)


def format_bdf(bdf: str) -> str:
    """'0000:46:00.0' -> '0000:46:00.0 (UBB2/U6)'. Falls back to raw bdf if unparseable."""
    pos = parse_bdf_physical(bdf)
    if pos is None:
        return bdf
    tray, asic = pos
    return f"{bdf} (UBB{tray}/U{asic})"


EXPECTED_CHIP_COUNT = 32


# Per-ASIC eth port classification on a BH Galaxy UBB.
# Keys are ASIC_LOCATION (1..8 within a UBB). Each value is the port-index
# bitmask for its connection type. Ports are ETH0..ETH11 (bits 0..11).
# Effective per-chip mask = bitmask & ENABLED_ETH (some ports are muxed off
# per chip — the table is the wiring intent, ENABLED_ETH is what FW actually
# brings up). User-confirmed mapping (outlogix / Adam C's org).
ETH_QSFP_BY_ASIC: dict[int, int] = {
    1: 0xC0F,  # ETH0-3, ETH10-11
    2: 0xC03,  # ETH0-1, ETH10-11
    3: 0xC03,  # ETH0-1, ETH10-11
    4: 0xC03,  # ETH0-1, ETH10-11
    5: 0xC0C,  # ETH2-3, ETH10-11
    6: 0xC00,  # ETH10-11
    7: 0xC00,  # ETH10-11
    8: 0xC00,  # ETH10-11
}
ETH_INTERNAL_BY_ASIC: dict[int, int] = {
    1: 0x3F0,  # ETH4-9
    2: 0x3FC,  # ETH2-9
    3: 0x3FC,  # ETH2-9
    4: 0x38C,  # ETH2-3, ETH7-9
    5: 0x073,  # ETH0-1, ETH4-6
    6: 0x07F,  # ETH0-6
    7: 0x07F,  # ETH0-6
    8: 0x00F,  # ETH0-3
}
# ExaMAX (passive + retimer combined — treated as one category per spec).
ETH_EXAMAX_BY_ASIC: dict[int, int] = {
    1: 0x000,
    2: 0x000,
    3: 0x000,
    4: 0x070,  # ETH4-6 (retimer)
    5: 0x380,  # ETH7-9 (passive)
    6: 0x380,  # ETH7-9 (passive)
    7: 0x380,  # ETH7-9 (passive)
    8: 0x3F0,  # ETH4-6 (retimer) + ETH7-9 (passive)
}

# BH Galaxy board ID prefixes → hardware revision.
# board_id source: tt-smi -s `device_info[].board_info.board_id` (hex string).
# Spec: BH Galaxy Revisions Comparison Table (Confluence SYS-4055).
# RevA/B silicon: GDDR runs at 14G, host-PCIe at Gen4.
# RevC silicon:   GDDR runs at 16G, host-PCIe at Gen5.
# Mixing the wrong expected values on the wrong rev is unsafe (RevC commands
# on RevA/B hardware can brick the board), so we gate the rev-dependent checks
# below and FAIL if the rev can't be unambiguously detected.
BOARD_ID_PREFIX_REVAB = "00000471"
BOARD_ID_PREFIX_REVC = "00000473"
BOARD_REV_EXPECTATIONS = {
    "RevA/B": {"gddr_speed": "14G", "pcie_gen": 4, "board_id_prefix": BOARD_ID_PREFIX_REVAB},
    "RevC": {"gddr_speed": "16G", "pcie_gen": 5, "board_id_prefix": BOARD_ID_PREFIX_REVC},
}
