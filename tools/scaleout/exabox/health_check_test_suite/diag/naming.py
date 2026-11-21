# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How a finding names a chip: its BDF, its tray (1-4) and its ASIC location (1-8)."""
from __future__ import annotations

from .galaxy import UBB_ETH_PORT_OF, parse_bdf_physical


def chip(bdf: str) -> dict:
    """``{"bdf", "tray", "asic"}`` for a BDF; tray and asic are None for one off the Galaxy map."""
    tray, asic = parse_bdf_physical(bdf) or (None, None)
    return {"bdf": bdf, "tray": tray, "asic": asic}


def blamed(bdfs) -> list[dict]:
    """``chip`` for each distinct BDF, in order: what a check stores in ``data["blamed_chips"]`` for the chips it blames."""
    return [chip(b) for b in dict.fromkeys(bdfs)]


def port_name(ptype: str, pid: int) -> str:
    """'QSFP P3' for the cage ports, 'TRACE P9' and the like for the built-in ones."""
    return f"{'QSFP' if ptype == 'QSFP_DD' else ptype} P{pid}"


def link_end(bdf: str, channel: int) -> dict:
    """One end of a link: the chip (see chip) with its Ethernet channel and the port the channel is wired to."""
    c = chip(bdf)
    port = UBB_ETH_PORT_OF.get((c["asic"], channel))
    return {**c, "channel": channel, "port": port_name(*port) if port else None}
