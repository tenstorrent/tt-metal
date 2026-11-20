# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How a finding names a chip: its BDF, its tray (1-4) and its ASIC location (1-8)."""
from __future__ import annotations

from .galaxy import parse_bdf_physical


def chip(bdf: str) -> dict:
    """``{"bdf", "tray", "asic"}`` for a BDF; tray and asic are None for one off the Galaxy map."""
    tray, asic = parse_bdf_physical(bdf) or (None, None)
    return {"bdf": bdf, "tray": tray, "asic": asic}
