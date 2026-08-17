# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper over the C++ scanner: build it on demand, run it, cache the result."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCANNER = HERE / "scan"


@dataclass(frozen=True)
class Site:
    index: int
    addr: int
    word: int
    op: str
    sfpu: bool

    def label(self) -> str:
        return f"{self.op}@0x{self.addr:05x}"


@dataclass(frozen=True)
class Scan:
    elf: str
    mode: str
    body_start: int
    body_end: int
    body_source: str
    cave_start: int
    cave_limit: int
    cave_source: str
    unpacker_mask: int
    fillers: dict
    sites: tuple

    def unpackers(self) -> tuple:
        """Which unpackers the kernel programs a datum count for. Empty means unknown."""
        return tuple(unit for unit in (0, 1) if self.unpacker_mask & (1 << unit))


_cache: dict = {}


def _build() -> None:
    subprocess.run(["make", "--silent", "scan"], cwd=HERE, check=True)


def scan(elf: str, mode: str = "sync") -> Scan:
    """Scan one thread ELF. Cached on (path, mtime, mode) so a sweep pays for it once."""
    elf = str(Path(elf).resolve())
    key = (elf, Path(elf).stat().st_mtime_ns, mode)
    if key in _cache:
        return _cache[key]

    # Always ask make: a binary built before a filler or opcode was added is worse
    # than a missing one, because it answers with a stale table instead of failing.
    _build()
    out = subprocess.run(
        [str(SCANNER), "--mode", mode, elf], capture_output=True, text=True, check=True
    ).stdout
    raw = json.loads(out)

    cave = raw["cave"] or {"start": 0, "limit": 0, "source": "none"}
    result = Scan(
        elf=elf,
        mode=raw["mode"],
        body_start=raw["body"]["start"],
        body_end=raw["body"]["end"],
        body_source=raw["body"]["source"],
        cave_start=cave["start"],
        cave_limit=cave["limit"],
        cave_source=cave["source"],
        unpacker_mask=raw["unpacker_mask"],
        fillers=raw["fillers"],
        sites=tuple(
            Site(
                index=site["index"],
                addr=site["addr"],
                word=site["word"],
                op=site["op"],
                sfpu=site["sfpu"],
            )
            for site in raw["sites"]
        ),
    )
    _cache[key] = result
    return result
