# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCIe AER counters per chip, read from sysfs, and the judgement of the new errors over a run."""
from __future__ import annotations

import re
import time
from contextlib import suppress
from pathlib import Path

from .model import FAIL, PASS, SKIP, WARN, Check
from .naming import blamed, chip

PCI_DEVICES = Path("/sys/bus/pci/devices")
TENSTORRENT_VENDOR = "0x1e52"
KINDS = {"corr": "aer_dev_correctable", "nonfatal": "aer_dev_nonfatal", "fatal": "aer_dev_fatal"}

# Correctable errors are normal background; they WARN only at this many new ones and this many per hour.
CORR_MIN = 2
CORR_PER_HOUR = 6.0
RATE_PER_MIN = 500  # corrected errors a minute on one root port that WARN as a weak link
BDF = re.compile(r"\b([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b")


def tt_devices(devices: Path = PCI_DEVICES) -> list[Path]:
    out = []
    for dev in sorted(devices.glob("*")):
        with suppress(OSError):
            if (dev / "vendor").read_text().strip() == TENSTORRENT_VENDOR:
                out.append(dev)
    return out


def read_aer(devices: Path = PCI_DEVICES) -> dict[str, dict[str, int]]:
    """``{bdf: {corr, nonfatal, fatal}}`` totals for every Tenstorrent device; a counter that can't be read is left out."""
    out: dict[str, dict[str, int]] = {}
    for dev in tt_devices(devices):
        counts = out[dev.name] = {}
        for kind, fname in KINDS.items():
            if (n := _total(dev / fname)) is not None:
                counts[kind] = n
    return out


def _total(path: Path) -> int | None:
    try:
        m = re.search(r"^TOTAL_ERR_\w+\s+(\d+)", path.read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) if m else None


def root_ports(devices: Path = PCI_DEVICES) -> dict[Path, list[Path]]:
    out: dict[Path, list[Path]] = {}
    for dev in tt_devices(devices):
        out.setdefault(dev.resolve().parent, []).append(dev)
    return out


def port_corrected(port: Path, chips: list[Path]) -> int | None:
    """A root port's corrected errors: the larger of aer_rootport_total_err_cor and its own and its chips'
    aer_dev_correctable (either can stay put while the other moves)."""
    own = [n for p in (port, *chips) if (n := _total(p / "aer_dev_correctable")) is not None]
    vals = [sum(own)] if own else []
    with suppress(OSError, ValueError, IndexError):
        vals.append(int((port / "aer_rootport_total_err_cor").read_text().split()[0]))
    return max(vals, default=None)


def kernel_events(lines) -> dict[str, list[int]]:
    """``{bdf: [corrected, uncorrected]}`` from kernel log lines: native 'PCIe Bus Error' lines, and firmware-first
    records ('device_id: <bdf>' with the record's 'event severity:'; those hosts leave the sysfs counters at 0)."""
    ev: dict[str, list[int]] = {}
    sev: dict[str, str] = {}
    for ln in lines:
        if "[Hardware Error]" in ln:
            tag = m.group(1) if (m := re.search(r"\{(\d+)\}\[Hardware Error\]", ln)) else ""
            if m := re.search(r"event severity: (\w+)", ln):
                sev[tag] = m.group(1).lower()
            elif m := re.search(r"device_id: " + BDF.pattern, ln):
                ev.setdefault(m.group(1), [0, 0])[sev.get(tag, "corrected") not in ("corrected", "info")] += 1
        elif "PCIe Bus Error" in ln and (m := BDF.search(ln)):
            ev.setdefault(m.group(1), [0, 0])["Uncorrect" in ln] += 1
    return ev


def aer_new(before: dict, after: dict) -> dict[str, dict[str, int]]:
    """New errors per chip between two reads. A counter that went down was restarted (the chip dropped off and came
    back): it all counts, and so does every counter of a chip that was not in *before*."""
    new = {}
    for bdf, now in after.items():
        was = before.get(bdf, {})
        delta = {k: n - was.get(k, 0) if n >= was.get(k, 0) else n for k, n in now.items() if n != was.get(k, 0)}
        if delta:
            new[bdf] = delta
    return new


def unread(*reads: dict) -> dict[str, list[str]]:
    """The counters of each chip that one of *reads* did not have (a chip missing from a read misses them all)."""
    chips = {b for r in reads for b in r}
    out = {b: [k for k in KINDS if any(k not in r.get(b, {}) for r in reads)] for b in sorted(chips)}
    return {b: ks for b, ks in out.items() if ks}


class AerWindow:
    """The AER errors over a whole run: mark() adds what is new since the last read. A reset can restart a chip's
    counts or keep them, so mark() before each reset and rebase() after it; check() at the end."""

    def __init__(self, read=read_aer):
        self.read, self.new, self.t0 = read, {}, time.time()
        self.first = self.last = read()

    def mark(self) -> None:
        now = self.read()
        for bdf, d in aer_new(self.last, now).items():
            tot = self.new.setdefault(bdf, {})
            for k, n in d.items():
                tot[k] = tot.get(k, 0) + n
        self.last = now

    def rebase(self) -> None:
        """Count from the counters as a reset left them."""
        self.last = self.read()

    def check(self) -> Check:
        self.mark()
        if not any(self.first.values()) and not any(self.last.values()):
            return Check(
                name="pcie_aer_delta",
                status=SKIP,
                details="no PCIe AER counter readable on any chip (AER off in the kernel or BIOS, or no chip on PCIe)",
                ip="pcie",
            )
        return aer_check(self.new, (time.time() - self.t0) / 3600, len(self.last), unread(self.first, self.last))


def judge(new: dict, hours: float) -> tuple[list, list]:
    """The chips with uncorrectable new errors, and those with correctable ones at CORR_MIN and CORR_PER_HOUR."""
    bad = [b for b, d in new.items() if d.get("nonfatal") or d.get("fatal")]
    return bad, [b for b, d in new.items() if CORR_MIN <= d.get("corr", 0) >= CORR_PER_HOUR * hours and b not in bad]


def aer_lines(before: dict, after: dict, hours: float, when: str) -> list[str]:
    """'AER FAIL 81:00.0 (tray 4): nonfatal +1 <when>': a line per chip whose new errors judge FAIL or WARN."""
    bad, noisy = judge(new := aer_new(before, after), hours)
    return [
        f"AER {'FAIL' if b in bad else 'WARN'} {b.split(':', 1)[-1]} (tray {chip(b)['tray']}): "
        + ", ".join(f"{k} +{n}" for k, n in new[b].items())
        + f" {when}"
        for b in [*bad, *noisy]
    ]


def aer_check(new: dict, hours: float, chips: int = 0, missing: dict | None = None) -> Check:
    """Uncorrectable new errors FAIL, correctable ones at CORR_MIN and CORR_PER_HOUR WARN, and so do counters
    *missing* at the start or the end of the window: errors on them were not counted."""
    missing = missing or {}
    bad, noisy = judge(new, hours)
    status = FAIL if bad else WARN if noisy or missing else PASS
    named = {b: new[b] for b in [*bad, *noisy]}
    not_read = [f"{b}: {', '.join(ks)} not read" for b, ks in missing.items()]
    return Check(
        name="pcie_aer_delta",
        status=status,
        details=(
            "; ".join([*(f"{b}: {d}" for b, d in named.items()), *not_read]) or f"no new AER errors over {chips} chips"
        ),
        data={
            "hours": hours,
            "new": [{**chip(b), **d} for b, d in new.items()],
            "flagged": [{**chip(b), **d} for b, d in named.items()],
            "missing": [{**chip(b), "counters": ks} for b, ks in missing.items()],
            "blamed_chips": blamed(named),
        },
        ip="pcie",
    )
