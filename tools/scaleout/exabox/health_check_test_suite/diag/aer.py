# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCIe AER counters per chip and root port, before and after the run (sysfs ``aer_dev_*``, no privileges)."""
from __future__ import annotations

from pathlib import Path

from .galaxy import format_bdf
from .model import FAIL, PASS, SKIP, WARN, Check, Phase
from .resets import chip_root_ports

PCIE_AER_DIFF = "pcie_aer_diff"  # the check this module emits


def _counters(dev: Path) -> dict[str, int]:
    """'<severity> <name>' -> count for one PCIe function; empty when it has no AER."""
    out = {}
    for sev in ("correctable", "nonfatal", "fatal"):
        try:
            text = (dev / f"aer_dev_{sev}").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            name, _, n = line.rpartition(" ")
            if name and not name.startswith("TOTAL_") and n.isdigit():
                out[f"{sev} {name}"] = int(n)
    return out


def read_aer(sys_root: Path = Path("/sys")) -> dict[str, dict[str, int]]:
    """chip BDF -> its own and its root port's counters, keyed 'chip <sev> <name>' and 'root port <sev> <name>'."""
    devices = sys_root / "bus" / "pci" / "devices"
    return {
        chip: {
            f"{side} {k}": n
            for side, bdf in (("chip", chip), ("root port", port))
            for k, n in _counters(devices / bdf).items()
        }
        for chip, port in chip_root_ports(sys_root).items()
    }


def check_aer(phase: Phase, before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> None:
    """Add ``pcie_aer_diff``: FAIL if an uncorrectable count rose, WARN if a correctable one did, SKIP with no counters."""
    if not any(before.values()):
        phase.add(Check(name=PCIE_AER_DIFF, status=SKIP, details="skipped: no AER counters in sysfs", ip="pcie"))
        return
    rose = {
        chip: {k: n - before[chip].get(k, 0) for k, n in now.items() if n > before[chip].get(k, 0)}
        for chip, now in after.items()
        if chip in before
    }
    rose = {chip: d for chip, d in rose.items() if d}
    uncorrectable = any(k.split()[-2] != "correctable" for d in rose.values() for k in d)  # sysfs names have no spaces
    phase.add(
        Check(
            name=PCIE_AER_DIFF,
            status=(FAIL if uncorrectable else WARN) if rose else PASS,
            details="; ".join(
                f"{format_bdf(chip)}: " + ", ".join(f"{k} +{n}" for k, n in sorted(d.items()))
                for chip, d in sorted(rose.items())
            )
            or f"{len(before)} chips and their root ports: no AER count rose during the run",
            data={"rose": rose} if rose else {},
            ip="pcie",
            parts=[format_bdf(chip) for chip in sorted(rose)],
        )
    )
