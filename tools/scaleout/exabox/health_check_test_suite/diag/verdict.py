# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The run's verdict: per fault where, its cause class and repair, then one 'Repair recommendation:' line.
ROWS names each known fault's checks by the emitting modules' constants (tests/test_diag_verdict.py)."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable, NamedTuple

from .aer import PCIE_AER_DIFF
from .busbar import (
    BUSBAR_BRICK_VOUT,
    BUSBAR_CHIP_LOST,
    BUSBAR_HANG,
    BUSBAR_HSC_TRIP,
    BUSBAR_NUMERIC,
    BUSBAR_PCIE_LINK,
    BUSBAR_QUIET_CLOCK,
    BUSBAR_REBOOT,
)
from .device_tests import (
    DIDT_MATMUL_GALAXY,
    GDDR_ACT_ALL,
    GDDR_ACT_C3,
    GDDR_FAST,
    GDDR_FULL,
    NOC_INTEGRITY,
    PCIE_READ,
    PCIE_WRITE,
)
from .eth_status import ETH_STATUS
from .eth_summary import _one_line
from .galaxy import parse_bdf_physical
from .model import FAIL, WARN
from .resets import POST_RESET_STATE_STABLE, POST_TEST_RESET_OK, RECOUNT, RESET
from .retest import RETEST_RESET_OK
from .sensors import TRAY_POWER
from .snapshot_checks import (
    ASIC_LOCATION_PER_UBB,
    ASIC_THERMAL_PRECHECK,
    DRAM_STATUS,
    GDDR_DBI_FW,
    GDDR_BIST_PER_CHANNEL,
    GDDR_TRAINING_PER_CHANNEL,
    PCIE_ENUM_COUNT,
    PCIE_GEN,
    PCIE_LANE_WIDTH,
)

if TYPE_CHECKING:
    from .sensors import TrayCause

# Cause classes: one is printed before the repair of every lost chip or tray.
CAUSES = ("POWER", "POWER-DOWNSTREAM", "PCIE/LINK", "CHIP/BOARD", "UNKNOWN")


class Row(NamedTuple):
    fault: str  # what the fault is, as people name it
    checks: tuple[str, ...]  # the checks that report it; "x*" is every check named x...
    cause: str  # the cause class printed before the repair ("" for none); a cause the reads set wins
    do: str  # the repair; "that tray" becomes the tray when every chip at fault is on one
    when: Callable[[dict], bool] | None = None  # a row with a test is tried before the rows without


def _trays(check: dict) -> Counter:
    """tray -> how many of the check's chips are on it."""
    return Counter(pos[0] for pos in map(parse_bdf_physical, check.get("parts") or []) if pos)


def _cause(want: str) -> Callable[[dict], bool]:
    return lambda c: c.get("cause") == want


def _says(words: str) -> Callable[[dict], bool]:
    return lambda c: words in (c.get("details") or "")


def _tests(*names: str) -> tuple[str, ...]:
    """Device test checks, and their retest after the reset (name + RETEST_SUFFIX)."""
    return tuple(n + "*" for n in names)


# Checks that find chips gone from the bus.
LOST = (PCIE_ENUM_COUNT, ASIC_LOCATION_PER_UBB, BUSBAR_CHIP_LOST)
RESET_LOST = (RESET + "*", RECOUNT + "*", POST_TEST_RESET_OK, RETEST_RESET_OK, POST_RESET_STATE_STABLE)
GDDR = _tests(GDDR_ACT_C3, GDDR_ACT_ALL, GDDR_FAST, GDDR_FULL) + (
    GDDR_BIST_PER_CHANNEL,
    GDDR_TRAINING_PER_CHANNEL,
    DRAM_STATUS,
)

_REGLOW = "tt-smi -glx_reset and recount"
_PCIE = "reseat that tray's host PCIe cable; swap it with a neighbour tray's to see if the fault follows"
_BOARD = "re-run; if it repeats on the same chip, replace that tray's board"

ROWS: dict[str, Row] = {
    "F01": Row(
        "tray loses 12 V under load",
        (TRAY_POWER, BUSBAR_HSC_TRIP),
        "POWER",
        "replace that tray's bus bar set, then that tray if it still drops",
        lambda c: c["status"] == FAIL and c.get("cause") == "POWER",
    ),
    "F02": Row(
        "HSC under-voltage latch",
        (TRAY_POWER,),
        "POWER",
        "watch that tray's 12 V feed; if its under-voltage latch sets again, replace its bus bar set",
        lambda c: c["status"] == WARN and c.get("cause") == "POWER",
    ),
    "F03": Row(
        "power brick output left at 12.00 V",
        (BUSBAR_BRICK_VOUT,),
        "POWER",
        "set every brick's output voltage to 12.24 V or 12.35 V, then re-run",
    ),
    "F04": Row(
        "bad bus bar",
        LOST + RESET_LOST,
        "POWER",
        f"fix that tray's 12 V feed: replace its bus bar set, then {_REGLOW}",
        _cause("POWER"),
    ),
    "F05": Row(
        "tray missing",
        LOST,
        "UNKNOWN",
        f"{_REGLOW}; if still missing, reseat that tray, then replace it",
        lambda c: not c.get("cause") and any(n >= 8 for n in _trays(c).values()),
    ),
    "F06": Row(
        "chip lost on reset",
        RESET_LOST,
        "CHIP/BOARD",
        f"{_REGLOW}; if it is lost on reset again, replace that tray's board",
    ),
    "F07": Row(
        "chip drops off PCIe with 12 V good",
        (TRAY_POWER,) + LOST + RESET_LOST,
        "POWER-DOWNSTREAM",
        "that tray's 12 V is good but a VR's power-good is low: replace that tray's board",
        _cause("POWER-DOWNSTREAM"),
    ),
    "F08": Row(
        "chip missing: PCIe link never trains",
        LOST,
        "PCIE/LINK",
        f"{_REGLOW}; if still missing, swap that tray's host PCIe cable with a neighbour tray's",
    ),
    "F09": Row(
        "PCIe correctable errors",
        (PCIE_AER_DIFF,),
        "PCIE/LINK",
        "reseat that tray's host PCIe cable; swap it with a neighbour tray's: if the errors follow the tray, "
        "replace that tray's board",
    ),
    "F10": Row("PCIe width or speed low", (PCIE_GEN, PCIE_LANE_WIDTH, BUSBAR_PCIE_LINK), "PCIE/LINK", _PCIE),
    "F11": Row(
        "PCIe bandwidth low",
        _tests(PCIE_READ, PCIE_WRITE),
        "PCIE/LINK",
        "check the link width and speed, then swap that tray's host PCIe cable with a neighbour tray's",
    ),
    "F13": Row(
        "Ethernet trace ports down",
        (ETH_STATUS + "*",),
        "CHIP/BOARD",
        "a trace port stays inside the tray: replace that tray's board",
        _says("TRACE port"),
    ),
    "F14": Row(
        "inter-host QSFP-DD link down or bad data",
        (ETH_STATUS + "*",),
        "PCIE/LINK",
        "reseat, then replace, the QSFP-DD cable on those ports",
        _says("QSFP-DD"),
    ),
    # F19 also covers numeric errors at Vmin (F17): same compare, same vdd_min repair.
    "F19": Row(
        "bf16/bf8 DIDT numeric errors",
        _tests(DIDT_MATMUL_GALAXY) + (BUSBAR_NUMERIC,),
        "CHIP/BOARD",
        "raise that chip's vdd_min to 760 mV and retest",
    ),
    "F21": Row("GDDR fault", GDDR, "CHIP/BOARD", _BOARD),
    "F22": Row(
        "GDDR DBI errors on old firmware",
        (GDDR_DBI_FW,),
        "",
        "flash firmware 19.17.1 or newer to every chip, then re-run",
    ),
    "F24": Row(
        "chip hangs under load",
        (BUSBAR_HANG,),
        "CHIP/BOARD",
        "tt-smi -glx_reset and re-run; if that chip hangs again, replace that tray's board",
    ),
    "F25": Row(
        "host rebooted under load", (BUSBAR_REBOOT,), "POWER", "check the host's PSUs and power feed against the SEL"
    ),
    "F26": Row(
        "chip clock stays high at idle",
        (BUSBAR_QUIET_CLOCK,),
        "",
        "flash the same firmware bundle to every chip and re-run; if it repeats, replace that tray's board",
    ),
    "F27": Row("overheating", (ASIC_THERMAL_PRECHECK,), "", "check that tray's fans and airflow"),
    "F29": Row("NOC data errors", _tests(NOC_INTEGRITY), "CHIP/BOARD", _BOARD),
}

# A check no row names: what to do by its IP block.
DO_BY_IP = {
    "pcie": _PCIE,
    "eth": "reseat or replace the cable on those ports; a trace port that stays down points at the tray board",
    "gddr": _BOARD,
    "fw": "flash the same firmware bundle to every chip",
    "thermal": "check that tray's fans and airflow",
}
DEFAULT_DO = "see the check's details and its log"


def _named(names: tuple[str, ...], name: str) -> bool:
    return any(name == n or (n.endswith("*") and name.startswith(n[:-1])) for n in names)


def row_of(check: dict) -> tuple[str, Row | None]:
    """(id, row) of the fault a WARN/FAIL check reports, else ("", None); rows with a test go first."""
    for tested in (True, False):
        for rid, row in ROWS.items():
            if (row.when is not None) == tested and _named(row.checks, check["name"]):
                if row.when is None or row.when(check):
                    return rid, row
    return "", None


def attribute_power(report: dict, causes: dict[int, TrayCause]) -> None:
    """Give each WARN/FAIL lost-chip check its tray's power cause, that reading first in its details."""
    for phase in report["phases"].values():
        for c in phase["checks"]:
            if c["status"] not in (FAIL, WARN) or c.get("cause") or not _named(LOST + RESET_LOST, c["name"]):
                continue
            trays = sorted(set(_trays(c)) & set(causes))
            if trays:
                c["cause"] = causes[trays[0]].cause
                c["details"] = "; ".join(causes[t].why for t in trays) + " — " + c["details"]


def _repair(check: dict, row: Row | None) -> str:
    """The repair, with 'that tray' made the tray when every chip at fault is on one."""
    do = row.do if row else DO_BY_IP.get(check.get("ip", "other"), DEFAULT_DO)
    trays = _trays(check)
    return do.replace("that tray", f"UBB{next(iter(trays))}") if len(trays) == 1 else do


def verdict_lines(report: dict) -> list[str]:
    """Fault, Cause and Repair lines per WARN/FAIL check of a gating phase (one per check and chips), the
    Evidence lines, the final verdict (INCOMPLETE when a gating phase errored and nothing failed), the repairs."""
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

    lines, repairs = [], {}
    for i, c in enumerate(faults, 1):
        rid, row = row_of(c)
        cause = c.get("cause") or (row.cause if row else "")
        do = _repair(c, row)
        what = f"{rid} {row.fault}: " if row else ""
        lines.append(
            f"Fault {i}: {c['status']} {what}{_one_line(c, 200)} | {'; '.join(c.get('parts') or []) or 'host'}"
        )
        if cause:
            lines.append(f"  Cause: {cause}")
        lines.append(f"  Repair: {do}")
        repairs.setdefault((c["status"] != FAIL, do), []).append(i)
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
    # FAIL repairs first; faults with the same repair share one entry.
    todo = [
        f"Fault {', '.join(map(str, ids))}: {do}" for (_, do), ids in sorted(repairs.items(), key=lambda kv: kv[0][0])
    ]
    return lines + [f"Final verdict: {final}", f"Repair recommendation: {'; '.join(todo) or 'none'}"]
