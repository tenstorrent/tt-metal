# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host chip identity from sysfs, forked from the galaxy-chip-id skill's chipid.py. Opens no device."""

from dataclasses import asdict, dataclass
from pathlib import Path

# tt-umd blackhole_implementation.hpp UBB_TRAY_BUS_IDS; tt-kmd galaxy_bdf_to_ordinal()
TRAY_OF_BUS_HIGH = {0x0: 1, 0x4: 2, 0xC: 3, 0x8: 4}


@dataclass
class Chip:
    dev: int
    bdf: str
    tray: int | None
    asic: int
    asic_id: str | None
    kmd: int

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _asic_id(node: Path) -> str | None:
    try:
        return f"0x{int((node / 'tt_asic_id').read_text().strip(), 16):016x}"
    except (OSError, ValueError):
        return None


def snapshot(sysfs: Path = Path("/sys/class/tenstorrent")) -> list[Chip]:
    """Chips sorted by bdf, dev = rank in that order: the device id a process sees with TT_VISIBLE_DEVICES unset.
    A missing chip shifts every later dev, so devices are joined on this snapshot, never on tray/asic arithmetic."""
    rows = []
    for node in sysfs.glob("tenstorrent!*"):
        bdf = (node / "device").resolve().name
        bus = int(bdf.split(":")[1], 16)
        rows.append((bdf, TRAY_OF_BUS_HIGH.get(bus >> 4), bus & 0xF, _asic_id(node), int(node.name.rsplit("!", 1)[1])))
    # bdf strings are fixed-width lowercase hex, so string order is bus order (= UMD's order).
    rows.sort()
    return [Chip(i, *row) for i, row in enumerate(rows)]


def by_dev(chips: list[Chip]) -> dict[int, Chip]:
    return {c.dev: c for c in chips}
