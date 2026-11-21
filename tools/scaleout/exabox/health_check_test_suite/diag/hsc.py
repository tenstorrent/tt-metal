# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Read-only tray power status, taken before and after the tests.

Each Galaxy tray's two hot-swap controllers (HSCs) latch an input under-voltage
flag when the tray's 12 V input sags, and an over-current fault when they trip
on the tray's current (usually turning the tray off). A weak bus bar usually
does not fail the test that sags it: the chip survives and the flag stays
latched. The tests phase reads STATUS_WORD and STATUS_INPUT of every HSC before
and after the load and reports a flag that was newly set during it.

Only those two PMBus registers are read, through the in-band BMC channel
(``ipmitool raw 0x06 0x52``). Nothing is written, and the latches are not
cleared, so a flag already set before the tests is reported as such rather than
judged.
"""
from __future__ import annotations

import os
import subprocess

from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# Tray number -> I2C bus of its HSCs, and the two HSC addresses on each.
HSC_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
HSC_ADDRS = (0x40, 0x42)

# PMBus registers read: the code and the number of bytes returned.
STATUS_WORD = (0x79, 2)
STATUS_INPUT = (0x7C, 1)

# STATUS_WORD bits (an HSC that turned itself off, an over-current trip) and
# STATUS_INPUT bits, in the order a reading names them.
WORD_BITS = {0x40: "OFF", 0x10: "IOUT_OC_FAULT"}
INPUT_BITS = {0x10: "VIN_UV_FAULT", 0x20: "VIN_UV_WARN"}
FLAG_ORDER = (*WORD_BITS.values(), *INPUT_BITS.values())

READ_TIMEOUT_S = 5

HscStatus = dict[tuple[int, int], "tuple[int, int] | None"]  # (tray, addr) -> (STATUS_WORD, STATUS_INPUT)


def _ipmitool() -> list[str]:
    return ["ipmitool"] if os.geteuid() == 0 else ["sudo", "-n", "ipmitool"]


def _read_register(tray: int, addr: int, reg: tuple[int, int]) -> int | None:
    code, nbytes = reg
    cmd = [
        *_ipmitool(),
        "raw",
        "0x06",
        "0x52",
        f"{HSC_BUS[tray]:#04x}",
        f"{addr:#04x}",
        f"{nbytes:#04x}",
        f"{code:#04x}",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=READ_TIMEOUT_S)
        if cp.returncode != 0:
            return None
        raw = [int(b, 16) for b in cp.stdout.split()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if len(raw) != nbytes:
        return None
    return int.from_bytes(bytes(raw), "little")


def read_hsc_status() -> HscStatus:
    """STATUS_WORD and STATUS_INPUT of every tray HSC; None for one that did not answer."""
    status: HscStatus = {}
    for tray in HSC_BUS:
        for addr in HSC_ADDRS:
            word = _read_register(tray, addr, STATUS_WORD)
            inp = _read_register(tray, addr, STATUS_INPUT)
            status[(tray, addr)] = None if word is None or inp is None else (word, inp)
    return status


def _flags(reading: "tuple[int, int]") -> set[str]:
    word, inp = reading
    return {name for bit, name in WORD_BITS.items() if word & bit} | {
        name for bit, name in INPUT_BITS.items() if inp & bit
    }


def flag_names(flags: set[str]) -> str:
    """The flags as one reason, e.g. 'OFF, IOUT_OC_FAULT' for an over-current trip."""
    return ", ".join(f for f in FLAG_ORDER if f in flags)


def judge_hsc(before: HscStatus, after: HscStatus, phase: Phase) -> None:
    """Add the ``hsc_input_status`` check: FAIL for a flag newly set during the
    tests, WARN for one already set before them or an HSC that was not read,
    SKIP when no HSC answered at all."""
    if not any(before.values()) and not any(after.values()):
        phase.add(
            Check(
                name="hsc_input_status",
                status=SKIP,
                details="no HSC answered (needs ipmitool and root or sudo -n); tray input under-voltage not judged",
                ip="board",
            )
        )
        return
    new, standing, unread = [], [], []
    for (tray, addr), post in after.items():
        pre = before.get((tray, addr))
        where = f"tray {tray} HSC {addr:#04x}"
        if post is None or pre is None:
            unread.append(f"{where} not read {'before' if pre is None else 'after'} the tests")
            continue
        if fresh := _flags(post) - _flags(pre):
            new.append(
                f"{where}: {flag_names(fresh)} newly set during the tests (STATUS_WORD {pre[0]:#06x} -> {post[0]:#06x}, STATUS_INPUT {pre[1]:#04x} -> {post[1]:#04x})"
            )
        if held := _flags(post) & _flags(pre):
            standing.append(
                f"{where}: {flag_names(held)} already set before the tests, so a new one cannot be told apart"
            )
    status = FAIL if new else WARN if standing or unread else PASS
    details = (
        "; ".join(new + standing + unread) or "no under-voltage, over-current or OFF flag on any of the 8 tray HSCs"
    )
    phase.add(
        Check(
            name="hsc_input_status",
            status=status,
            details=details,
            data={
                "before": {f"{t}:{a:#04x}": v for (t, a), v in before.items()},
                "after": {f"{t}:{a:#04x}": v for (t, a), v in after.items()},
            },
            ip="board",
        )
    )
