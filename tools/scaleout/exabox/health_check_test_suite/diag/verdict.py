# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The run's verdict: one line per fault saying what, where and what to do,
then one final line. Printed last, so the end of the log says what needs fixing."""
from __future__ import annotations

from .eth_summary import _one_line
from .galaxy import parse_bdf_physical
from .model import FAIL, WARN

_LOST = "tt-smi -glx_reset and recount; if still missing, check that tray's 12 V and its host PCIe cable"

# What to do about a fault, by the check that found it, else by its IP block.
DO_BY_CHECK = {
    "pcie_enum_count": _LOST,
    "asic_location_per_ubb": _LOST,
    "post_test_reset_ok": _LOST,
}
DO_BY_IP = {
    "pcie": "reseat that tray's host PCIe cable; swap it with a neighbour tray's to see if the fault follows",
    "eth": "reseat or replace the cable on those ports; a trace port that stays down points at the tray board",
    "gddr": "re-run; a fault that repeats on the same chip points at that tray board",
    "fw": "flash the same firmware bundle to every chip",
    "thermal": "check that tray's fans and airflow",
}
DEFAULT_DO = "see the check's details and its log"


def _do(check: dict) -> str:
    """What to do, with 'that tray' made the tray when every chip at fault is on one."""
    if check["name"].startswith("reset_"):
        do = _LOST
    else:
        do = DO_BY_CHECK.get(check["name"]) or DO_BY_IP.get(check.get("ip", "other"), DEFAULT_DO)
    trays = {pos[0] for pos in map(parse_bdf_physical, check.get("parts") or []) if pos}
    return do.replace("that tray", f"UBB{trays.pop()}") if len(trays) == 1 else do


def verdict_lines(report: dict) -> list[str]:
    """``Fault i: what | where | do`` per WARN/FAIL check of a gating phase,
    ``Evidence:`` per capture of note, then ``Final verdict: ...``.

    The same check naming the same chips in several snapshots is one fault.
    A gating phase that errored with no FAIL leaves the run INCOMPLETE: the
    unit was not shown faulty, only not fully tested.
    """
    faults, seen, incomplete = [], set(), []
    for name, phase in report["phases"].items():
        if not phase.get("gates", True):
            continue
        error = phase.get("error")
        if error:
            incomplete.append(name)
        for c in phase["checks"]:
            if error and c["details"] == error:
                continue  # the crash itself: INCOMPLETE below, not a fault of the unit
            key = (c["name"], c["status"], tuple(c.get("parts") or ()))
            if c["status"] in (FAIL, WARN) and key not in seen:
                seen.add(key)
                faults.append(c)

    lines = [
        f"Fault {i}: {c['status']} {_one_line(c, 200)} | {'; '.join(c.get('parts') or []) or 'host'} | {_do(c)}"
        for i, c in enumerate(faults, 1)
    ]
    captures = report["phases"].get("captures", {}).get("checks", [])
    lines += [f"Evidence: {_one_line(c, 200)}" for c in captures if c["status"] == WARN]

    n_fail = sum(c["status"] == FAIL for c in faults)
    if n_fail:
        final = f"FAIL: {n_fail} fault(s)" + (f", {len(faults) - n_fail} warning(s)" if len(faults) > n_fail else "")
    elif incomplete:
        final = f"INCOMPLETE (not a fault): {', '.join(incomplete)} did not finish"
    elif faults:
        final = f"WARN: {len(faults)} warning(s)"
    else:
        final = "PASS"
    return lines + [f"Final verdict: {final}"]
