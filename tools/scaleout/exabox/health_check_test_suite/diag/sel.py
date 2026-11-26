# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The BMC's system event log (SEL), read in-band before the tests (the last 7 days) and after the end sequence (the
new events). Without root or a sudo rule for the read, a SKIP that says why. The BMC names tray N's sensors UBB<N-1>."""
from __future__ import annotations

import re
import time
from collections import Counter

from .common import root_cmd, run_with_timeout
from .model import FAIL, PASS, SKIP, WARN, Check

SEL_CMD = ["ipmitool", "sel", "elist"]
SEL_TIMEOUT_S = 120
HISTORY_S = 7 * 86400
BUS_CORRECTABLE_WARN, FRB2_WARN = 20, 3
NEW_EVENT = re.compile(r"FRB|Presence|PresDet|Power Unit|Power Supply|PSU|Current|Voltage|Critical|Temp|Thermal", re.I)
THERMAL_TRIP = re.compile(r"Upper (Critical|Non-recoverable)|Thermal Trip|Overtemp", re.I)


def read_sel() -> tuple[list[str] | None, str]:
    """The SEL's rows, or None and why it could not be read."""
    cmd, why = root_cmd(SEL_CMD)
    if cmd is None:
        return None, why
    rc, out, err, _ = run_with_timeout(cmd, SEL_TIMEOUT_S)
    return (out.splitlines(), "") if rc == 0 else (None, f"{' '.join(SEL_CMD)} rc {rc}: {err.strip()[-200:]}")


def _tray(sensor: str) -> str:
    m = re.search(r"UBB(\d)", sensor)
    return f"tray {int(m.group(1)) + 1}" if m else "host"


def _rows(lines: list[str]):
    """(epoch, sensor, event, state, row) for each row with a readable date."""
    for row in lines:
        p = [x.strip() for x in row.split("|")]
        if len(p) < 6:
            continue
        try:
            t = time.mktime(time.strptime(f"{p[1]} {p[2]}", "%m/%d/%Y %H:%M:%S"))
        except ValueError:
            continue
        yield t, (p[3].split() or ["?"])[-1], p[4], p[5], row


def _check(name: str, status: str, findings: list[str], ok: str) -> Check:
    status = status if findings else PASS
    return Check(name=name, status=status, details="; ".join(findings) or ok, data={"findings": findings}, ip="board")


def history_check(lines: list[str] | None, why: str, now: float) -> Check:
    """In the HISTORY_S before *now*: many bus correctable errors, an FRB3, repeated FRB2 hangs in POST and critical
    threshold crossings, each a WARN naming its tray."""
    if lines is None:
        return Check(name="sel_history", status=SKIP, details=f"SEL not read: {why}", ip="board")
    n: Counter = Counter()
    for t, sensor, event, _state, _row in _rows(lines):
        if now - t > HISTORY_S:
            continue
        if "Bus Correctable" in event:
            n["host", "Bus Correctable errors"] += 1
        elif "FRB3" in event:
            n[_tray(sensor), "FRB3 processor init failures"] += 1
        elif "FRB2" in event:
            n[_tray(sensor), "FRB2 hangs in POST"] += 1
        elif "Critical going" in event:
            n[_tray(sensor), f"{sensor} critical threshold crossings"] += 1
    warn = {"Bus Correctable errors": BUS_CORRECTABLE_WARN, "FRB2 hangs in POST": FRB2_WARN}
    found = [f"{where}: {k} {what} in 7 days" for (where, what), k in sorted(n.items()) if k >= warn.get(what, 1)]
    return _check("sel_history", WARN, found, "no SEL events to flag in 7 days")


def new_check(before: list[str] | None, after: list[str] | None, why: str) -> Check:
    """The rows new since *before*: a temperature sensor's thermal trip FAILs its tray, any other row about FRB,
    presence, power, current, voltage or temperature is a WARN naming its tray."""
    if before is None or after is None:
        return Check(name="sel_new", status=SKIP, details=f"SEL not read: {why}", ip="board")
    old, trips, others = set(before), [], []
    for _t, sensor, event, state, _row in _rows([r for r in after if r not in old and NEW_EVENT.search(r)]):
        if "Deasserted" not in state:
            trip = re.search(r"Temp|Thermal|TMAX", sensor, re.I) and THERMAL_TRIP.search(f"{event} {state}")
            (trips if trip else others).append(f"{_tray(sensor)}: {sensor} {event} {state}".strip())
    return _check(
        "sel_new", FAIL if trips else WARN, [f"thermal trip {x}" for x in trips] + others, "no new SEL events to flag"
    )
