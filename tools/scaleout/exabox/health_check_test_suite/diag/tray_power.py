# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 1b: tray 12 V feed latches (hot-swap controllers and tray CPLD).

A failing bus bar or tray 12 V feed sags a tray's input under load, and the tray
drops off PCIe as 8 missing chips with nothing pointing at the feed. Each tray's
two MP5990 hot-swap controllers (HSCs) latch what they saw: STATUS_WORD OFF and
IOUT_OC, STATUS_IOUT IOUT_OC_FAULT, STATUS_INPUT VIN_UV_FAULT and VIN_UV_WARN.
The tray CPLD's reg 0x1E reads ff while the tray is up. Together they tell a
tray lost to its power feed from one lost to a chip or board fault.

The latches hold until the tray is re-powered (a power cycle or a glx_reset), so
the phase runs on pre_bmc_reboot and post_bmc_reboot only (neither resets the
unit), right after the snapshot.

Reads only, in-band through the BMC's I2C master (`ipmitool raw 0x06 0x52`).
Every transaction is checked against READS before ipmitool runs; nothing is
ever written. The phase does not gate: VIN_UV_WARN also latches on feeds that
sag without dropping the tray.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from .common import log, run
from .galaxy import TRAY_TO_UBB_NIBBLE
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# Tray -> its I2C bus behind the BMC. HSC and CPLD addresses are 8-bit, as ipmitool takes them.
TRAY_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
HSC_ADDRS = (0x40, 0x42)
CPLD_ADDR = 0x80
STATUS_WORD, STATUS_IOUT, STATUS_INPUT = 0x79, 0x7B, 0x7C
CPLD_TRAY_ALIVE = 0x1E
CPLD_ALIVE = 0xFF  # reg 0x1E while the tray is up

# (addr, command code) -> bytes read back: every transaction the phase may send.
READS = {
    **{(a, STATUS_WORD): 2 for a in HSC_ADDRS},
    **{(a, STATUS_IOUT): 1 for a in HSC_ADDRS},
    **{(a, STATUS_INPUT): 1 for a in HSC_ADDRS},
    (CPLD_ADDR, CPLD_TRAY_ALIVE): 1,
}

TRAY_POWER_HINT = "bus bar / tray 12 V feed"
CALL_TIMEOUT_S = 15
# Whole-phase bound: an unresponsive BMC must not eat the run's own timeout.
PHASE_TIMEOUT_S = 120
# Timeouts on one tray's bus before any read of it succeeded: skip the rest of
# that tray. Every tray's bus doing so, with no read ok, means no route: stop.
NO_ROUTE_TIMEOUTS = 2
# The CPLD bus is shared with the BMC's own polling: read until two in a row agree.
CPLD_MAX_READS = 4
# stderr phrases of a real sudo refusal (no route); a sudo warning line is not one.
SUDO_REFUSALS = (
    "a password is required",
    "a terminal is required",
    "is not in the sudoers",
    "is not allowed to execute",
    "sudo: ipmitool: command not found",
)


def tray_bdf_range(tray: int) -> str:
    n = TRAY_TO_UBB_NIBBLE[tray]
    return f"{n:x}1:00.0-{n:x}8:00.0"


def hsc_faults(word: int, inp: int, iout: int | None) -> list[str]:
    """Latched faults of one HSC. CML and the other status bits are not faults."""
    flags = (
        ("OFF", word & 0x40),
        ("IOUT_OC", word & 0x10 or (iout or 0) & 0x80),
        ("VIN_UV_FAULT", inp & 0x10),
        ("VIN_UV_WARN", inp & 0x20),
    )
    return [name for name, on in flags if on]


class TrayPowerReader:
    """In-band reads of the tray HSCs and CPLDs, gated by READS.

    `runner(argv) -> CompletedProcess` and `clock` are injectable so the tests
    see every transaction without ipmitool. Once the phase time is spent, or
    there is no route, nothing more is sent: `stopped` says why and every later
    read returns None.
    """

    def __init__(self, runner=None, clock=time.monotonic):
        self.base = ["ipmitool"] if os.geteuid() == 0 else ["sudo", "-n", "ipmitool"]
        self.clock = clock
        self.deadline = clock() + PHASE_TIMEOUT_S
        self.runner = runner or self._run
        self.reads_ok = 0
        self.bus_ok: dict[int, int] = {}
        self.bus_timeouts: dict[int, int] = {}
        self.bus_err: dict[int, str] = {}  # last error per bus: the 'not read' reason of its tray
        self.first_err = ""
        self.stopped = ""

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        left = self.deadline - self.clock()
        return run(argv, capture_output=True, text=True, timeout=max(1.0, min(CALL_TIMEOUT_S, left)))

    def _err(self, bus: int, msg: str) -> None:
        self.first_err = self.first_err or msg
        self.bus_err[bus] = msg

    def txn(self, bus: int, addr: int, code: int) -> int | None:
        """One register read (low byte first), or None if it was not read."""
        nbytes = READS.get((addr, code))
        if bus not in TRAY_BUS.values() or nbytes is None:
            raise ValueError(f"tray power: refusing bus 0x{bus:02x} addr 0x{addr:02x} code 0x{code:02x}")
        if not self.stopped and self.clock() >= self.deadline:
            self.stopped = f"phase time limit of {PHASE_TIMEOUT_S:g} s spent"
        if self.stopped or (self.bus_timeouts.get(bus, 0) >= NO_ROUTE_TIMEOUTS and not self.bus_ok.get(bus)):
            return None
        argv = self.base + ["raw", "0x06", "0x52"] + [f"0x{b:02x}" for b in (bus, addr, nbytes, code)]
        try:
            cp = self.runner(argv)
        except subprocess.TimeoutExpired as e:
            self.bus_timeouts[bus] = self.bus_timeouts.get(bus, 0) + 1
            self._err(bus, f"ipmitool: no reply in {e.timeout:g} s")
            return None
        except FileNotFoundError as e:
            self._err(bus, repr(e))
            self.stopped = "no route: ipmitool not found"
            return None
        except OSError as e:
            self._err(bus, repr(e))
            return None
        if cp.returncode != 0:
            lines = [ln.strip() for ln in (cp.stderr or cp.stdout or "").splitlines() if ln.strip()]
            refusal = next((ln for ln in lines if any(m in ln.lower() for m in SUDO_REFUSALS)), None)
            if refusal:
                self._err(bus, refusal)
                self.stopped = f"no route: {refusal}"
                return None
            # sudo may warn first ('sudo: unable to resolve host ...') and still run
            # ipmitool: keep ipmitool's own error, and read the other trays.
            err = next((ln for ln in lines if not ln.lower().startswith("sudo:")), None)
            self._err(bus, err or (lines[0] if lines else f"rc {cp.returncode}"))
            return None
        toks = (cp.stdout or "").split()
        if len(toks) != nbytes or not all(re.fullmatch(r"[0-9a-fA-F]{2}", t) for t in toks):
            self._err(bus, f"unparsed reply {cp.stdout!r}")
            return None
        self.reads_ok += 1
        self.bus_ok[bus] = self.bus_ok.get(bus, 0) + 1
        return int.from_bytes(bytes(int(t, 16) for t in toks), "little")

    def hsc_status(self, bus: int, addr: int) -> dict:
        """One HSC's STATUS_WORD, STATUS_INPUT and STATUS_IOUT (None where not read). One retry per read."""
        regs = (("status_word", STATUS_WORD), ("status_input", STATUS_INPUT), ("status_iout", STATUS_IOUT))
        out = {}
        for key, code in regs:
            v = self.txn(bus, addr, code)
            out[key] = v if v is not None else self.txn(bus, addr, code)
        return out

    def cpld_alive(self, bus: int) -> int | None:
        """CPLD reg 0x1E once two reads in a row agree, else None."""
        prev = None
        for _ in range(CPLD_MAX_READS):
            v = self.txn(bus, CPLD_ADDR, CPLD_TRAY_ALIVE)
            if v is not None and v == prev:
                return v
            prev = v
        return None


def tray_power_check(tray: int, hscs: dict, alive: int | None, why: str = "") -> Check:
    """One tray's verdict from its HSC reads ({addr: hsc_status()}) and CPLD reg 0x1E.

    FAIL, cause POWER: an HSC latched a fault; the tray's feed is the suspect.
    WARN, cause CHIP/BOARD: the tray is down with clean HSC latches: not the feed.
    WARN: a read is missing (`why` says why reads were cut short). Else PASS.
    """
    where = f"tray {tray} ({tray_bdf_range(tray)})"
    down = alive is not None and alive != CPLD_ALIVE
    data = {"tray": tray, "bdf_range": tray_bdf_range(tray), "hsc": {}, "cpld_reg1e": alive, "cause": None}
    findings, unread = [], []
    for addr, st in sorted(hscs.items()):
        key = f"0x{addr:02x}"
        word, inp, iout = st["status_word"], st["status_input"], st["status_iout"]
        if word is None or inp is None:
            unread.append(f"HSC {key}")
            data["hsc"][key] = dict(st, faults=None)
            continue
        faults = hsc_faults(word, inp, iout)
        data["hsc"][key] = dict(st, faults=faults)
        if faults:
            iout_txt = "not read" if iout is None else f"0x{iout:02x}"
            findings.append(
                f"HSC {key}: {', '.join(faults)} latched "
                f"(STATUS_WORD 0x{word:04x}, STATUS_INPUT 0x{inp:02x}, STATUS_IOUT {iout_txt})"
            )
    if alive is None:
        unread.append("CPLD")
    head = where
    if findings:
        data["cause"] = "POWER"
        head = f"{where}: POWER: check {TRAY_POWER_HINT}"
    elif down and not unread:
        data["cause"] = "CHIP/BOARD"
        head = f"{where}: CHIP/BOARD: tray down with no 12 V feed latch set, not the feed"
    body = findings + ([f"CPLD reg 0x1E=0x{alive:02x}: tray down"] if down else [])
    if unread:
        body.append(f"not read: {', '.join(unread)}" + (f" ({why})" if why else ""))
    if not body:
        status, lines = PASS, [f"{where}: no 12 V latch set, tray up"]
    else:
        status = FAIL if findings else WARN
        lines = [head] + body if data["cause"] else [f"{where}: {body[0]}"] + body[1:]
    return Check(name=f"tray{tray}_12v_feed", status=status, details="\n".join(lines), data=data, ip="board")


def run_tray_power(phase: Phase, dry_run: bool, runner=None, clock=time.monotonic) -> None:
    """Read and judge every tray's 12 V feed latches into `phase`.

    Every tray is tried, so a dead tray does not hide the others. Trays left
    when PHASE_TIMEOUT_S is spent are WARN 'not read'. No successful read on any
    tray means no route: one SKIP.
    """
    if dry_run:
        phase.add(Check(name="tray_12v_feed", status=SKIP, details="(dry) ipmitool raw 0x06 0x52 reads", ip="board"))
        return
    reader = TrayPowerReader(runner, clock)
    checks = []
    for tray, bus in sorted(TRAY_BUS.items()):
        hscs = {addr: reader.hsc_status(bus, addr) for addr in HSC_ADDRS}
        alive = reader.cpld_alive(bus)
        checks.append(tray_power_check(tray, hscs, alive, why=reader.stopped or reader.bus_err.get(bus, "")))
    if reader.stopped:
        log(f"tray_power: stopped reading: {reader.stopped}")
    if reader.reads_ok == 0:
        # No route at all (no ipmitool, no sudo, no BMC): one SKIP, not four WARNs.
        phase.add(
            Check(name="tray_12v_feed", status=SKIP, details=f"not read: {reader.first_err or 'no reply'}", ip="board")
        )
        return
    for c in checks:
        phase.add(c)
