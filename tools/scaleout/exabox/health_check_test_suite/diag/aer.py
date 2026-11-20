# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCIe AER counters per chip, read from sysfs, and the judgement of a before/after pair."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .model import FAIL, PASS, WARN, Check
from .naming import chip

PCI_DEVICES = Path("/sys/bus/pci/devices")
TENSTORRENT_VENDOR = "0x1e52"
KINDS = {"corr": "aer_dev_correctable", "nonfatal": "aer_dev_nonfatal", "fatal": "aer_dev_fatal"}

# Correctable errors are normal background; they WARN only at this many new ones and this many per hour.
CORR_MIN = 2
CORR_PER_HOUR = 6.0


def read_aer(devices: Path = PCI_DEVICES) -> dict[str, dict[str, int]]:
    """``{bdf: {corr, nonfatal, fatal}}`` totals for every Tenstorrent device; a counter that can't be read is left out."""
    out: dict[str, dict[str, int]] = {}
    for dev in sorted(devices.glob("*")):
        try:
            if (dev / "vendor").read_text().strip() != TENSTORRENT_VENDOR:
                continue
        except OSError:
            continue
        counts = out[dev.name] = {}
        for kind, fname in KINDS.items():
            try:
                m = re.search(r"^TOTAL_ERR_\w+\s+(\d+)", (dev / fname).read_text(), re.M)
            except OSError:
                continue
            if m:
                counts[kind] = int(m.group(1))
    return out


def load_aer_baseline(report: Path) -> tuple[dict, float]:
    """The counters and the time (epoch s) of the pcie_aer_delta read in an earlier diag_report.json."""
    checks = json.loads(report.read_text())["phases"]["triage"]["checks"]
    data = next(c["data"] for c in checks if c["name"] == "pcie_aer_delta")
    return data["after"], data["epoch"]


def aer_check(before: dict, after: dict, hours: float) -> Check:
    """New AER errors per chip between two reads: uncorrectable FAIL, correctable at CORR_MIN and CORR_PER_HOUR WARN."""
    new = {}
    for bdf, now in after.items():
        was = before.get(bdf, {})
        delta = {k: now[k] - was[k] for k in now if k in was and now[k] > was[k]}
        if delta:
            new[bdf] = delta
    bad = {b: d for b, d in new.items() if d.get("nonfatal") or d.get("fatal")}
    noisy = {
        b: d
        for b, d in new.items()
        if d.get("corr", 0) >= CORR_MIN and d["corr"] >= CORR_PER_HOUR * hours and b not in bad
    }
    status = FAIL if bad else WARN if noisy else PASS
    named = {b: new[b] for b in [*bad, *noisy]}
    return Check(
        name="pcie_aer_delta",
        status=status,
        details=(
            "; ".join(f"{b}: {d}" for b, d in named.items()) or f"no new AER errors over {len(after)} chips"
        ),
        data={
            "epoch": time.time(),
            "hours": hours,
            "before": before,
            "after": after,
            "new": [{**chip(b), **d} for b, d in new.items()],
            "flagged": [{**chip(b), **d} for b, d in named.items()],
        },
        ip="pcie",
    )
