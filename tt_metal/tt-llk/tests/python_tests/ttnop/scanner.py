# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper over the C++ scanner"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = os.environ.get("CHIP_ARCH", "wormhole").strip().lower()
SCANNERS = {arch: HERE / f"scan-{arch}" for arch in ("wormhole", "blackhole", "quasar")}
SCANNER = SCANNERS[ARCH]
SCANNER_ELF_NAME = "kernel.elf"
TTNOP_ELF_NAMES = {
    name: SCANNER_ELF_NAME
    for name in ("unpack.elf", "math.elf", "pack.elf", "sfpu.elf")
}
SCAN_COMMANDS = {
    "sync": (str(SCANNER), "--mode", "sync", SCANNER_ELF_NAME),
    "all": (str(SCANNER), "--mode", "all", SCANNER_ELF_NAME),
}


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
    cave_start: int
    cave_limit: int
    unpacker_mask: int
    fillers: dict
    sites: tuple

    def unpackers(self) -> tuple:
        """Which unpackers the kernel programs a datum count for. Empty means unknown."""
        return tuple(unit for unit in (0, 1) if self.unpacker_mask & (1 << unit))


_cache: dict = {}


def _build() -> None:
    subprocess.run(
        ["make", "--silent", SCANNER.name], cwd=HERE, check=True, shell=False
    )


def scan(elf: str, mode: str = "sync") -> Scan:
    """Scan one thread ELF. Cached on (path, mtime, mode) so a sweep pays for it once."""
    command = SCAN_COMMANDS[mode]
    elf_path = Path(elf).resolve()
    build_root = (
        Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "tt-llk-build"
    ).resolve()
    elf_path.relative_to(build_root)
    safe_elf_name = TTNOP_ELF_NAMES[elf_path.name]
    elf = str(elf_path)
    key = (elf, elf_path.stat().st_mtime_ns, mode)
    if key in _cache:
        return _cache[key]

    _build()
    with tempfile.TemporaryDirectory(prefix="ttnop-scan-") as temp_dir:
        shutil.copyfile(elf_path, Path(temp_dir) / safe_elf_name)
        out = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        ).stdout
    raw = json.loads(out)

    cave = raw["cave"] or {"start": 0, "limit": 0}
    result = Scan(
        elf=elf,
        mode=raw["mode"],
        cave_start=cave["start"],
        cave_limit=cave["limit"],
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
