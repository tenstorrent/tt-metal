# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmarks for the ternary engine, with a shared run record.

Every number this project reports goes through :class:`RunRecord`, which pins
the commit, chip, env, firmware and exact command next to the measurement, so a
number can always be traced back to how it was produced.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.record --list
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Chip peak DRAM bandwidth, GB/s (p300c, per chip).
DRAM_PEAK_GBS = 512.0


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def firmware_version() -> str:
    """The tt-metal firmware/fw bundle version, if this checkout has one."""
    for p in ("tt_metal/firmware/hex.file", "tt_metal/firmware/version.txt"):
        path = Path(os.environ.get("TT_METAL_HOME", ".")) / p
        if path.exists():
            return path.read_text().strip()[:64]
    return _sh("cd ${TT_METAL_HOME:-.} && git rev-parse --short HEAD")


@dataclass
class RunRecord:
    """One measurement plus everything needed to reproduce it.

    ``samples`` keeps every individual timing so the median is auditable and a
    later reader can see the spread, not just the central value.
    """

    name: str
    samples: list[float] = field(default_factory=list)  # seconds
    meta: dict = field(default_factory=dict)

    @property
    def median(self) -> float:
        return float(np.median(self.samples)) if self.samples else float("nan")

    @property
    def minimum(self) -> float:
        return float(np.min(self.samples)) if self.samples else float("nan")

    @property
    def maximum(self) -> float:
        return float(np.max(self.samples)) if self.samples else float("nan")

    def gbs(self, nbytes: int) -> float:
        m = self.median
        return nbytes / m / 1e9 if m > 0 else 0.0

    def frac_peak(self, nbytes: int) -> float:
        return self.gbs(nbytes) / DRAM_PEAK_GBS

    def summary(self, nbytes: int | None = None) -> str:
        base = (
            f"{self.name:28s} median {self.median * 1e6:9.2f} us  "
            f"[{self.minimum * 1e6:.2f}, {self.maximum * 1e6:.2f}]  n={len(self.samples)}"
        )
        if nbytes:
            base += f"  {self.gbs(nbytes):7.1f} GB/s  {self.frac_peak(nbytes) * 100:5.1f}% peak"
        return base

    def to_dict(self, nbytes: int | None = None) -> dict:
        d = {
            "name": self.name,
            "samples_s": self.samples,
            "median_us": self.median * 1e6,
            "min_us": self.minimum * 1e6,
            "max_us": self.maximum * 1e6,
            "meta": self.meta,
        }
        if nbytes:
            d["bytes"] = nbytes
            d["gbs"] = self.gbs(nbytes)
            d["frac_peak"] = self.frac_peak(nbytes)
        return d


def environment() -> dict:
    """Commit, chip, env, firmware and host: what every reported number needs."""
    home = os.environ.get("TT_METAL_HOME", ".")
    return {
        "when": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "commit": _sh(f"git -C {home} rev-parse HEAD"),
        "commit_short": _sh(f"git -C {home} rev-parse --short HEAD"),
        "branch": _sh(f"git -C {home} rev-parse --abbrev-ref HEAD"),
        "dirty": bool(_sh(f"git -C {home} status --porcelain")),
        "firmware": firmware_version(),
        "tt_metal_home": home,
        "chip": os.environ.get("TT_VISIBLE_DEVICES"),
        "mesh": os.environ.get("MESH_DEVICE"),
        "tt_metal_cache": os.environ.get("TT_METAL_CACHE"),
        "tt_cache_path": os.environ.get("TT_CACHE_PATH"),
        "argv": sys.argv,
        "python": platform.python_version(),
    }


def new_record(name: str, **extra) -> RunRecord:
    meta = environment()
    meta.update(extra)
    return RunRecord(name=name, meta=meta)


def append_jsonl(path, record: RunRecord, nbytes: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict(nbytes)) + "\n")
