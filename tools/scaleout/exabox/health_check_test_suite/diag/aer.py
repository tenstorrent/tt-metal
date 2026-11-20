# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCIe AER counters per chip, read from sysfs, and the judgement of the new errors over a run."""
from __future__ import annotations

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


def aer_new(before: dict, after: dict) -> dict[str, dict[str, int]]:
    """New errors per chip between two reads. A counter that went down was restarted by a reset: it all counts."""
    new = {}
    for bdf, now in after.items():
        was = before.get(bdf, {})
        delta = {k: now[k] - was[k] if now[k] >= was[k] else now[k] for k in now if k in was and now[k] != was[k]}
        if delta:
            new[bdf] = delta
    return new


class AerWindow:
    """The AER errors over a whole run. A reset restarts the kernel's counts, so each mark() adds what is new since
    the last one; mark before every reset and at the end."""

    def __init__(self, read=read_aer):
        self.read, self.last, self.new, self.t0 = read, read(), {}, time.time()

    def mark(self) -> None:
        now = self.read()
        for bdf, d in aer_new(self.last, now).items():
            tot = self.new.setdefault(bdf, {})
            for k, n in d.items():
                tot[k] = tot.get(k, 0) + n
        self.last = now

    def check(self) -> Check:
        self.mark()
        return aer_check(self.new, (time.time() - self.t0) / 3600, len(self.last))


def aer_check(new: dict, hours: float, chips: int = 0) -> Check:
    """Uncorrectable new errors FAIL, correctable ones at CORR_MIN and CORR_PER_HOUR WARN."""
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
        details=("; ".join(f"{b}: {d}" for b, d in named.items()) or f"no new AER errors over {chips} chips"),
        data={
            "hours": hours,
            "new": [{**chip(b), **d} for b, d in new.items()],
            "flagged": [{**chip(b), **d} for b, d in named.items()],
        },
        ip="pcie",
    )
