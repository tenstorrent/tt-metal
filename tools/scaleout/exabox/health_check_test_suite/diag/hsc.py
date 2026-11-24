# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tray power reads through the in-band BMC channel (``ipmitool raw``): each Galaxy tray's two hot-swap controllers
(HSCs), each tray CPLD's power-good byte, the BMC's power sensors and, opt-in, the power bricks behind their I2C mux.

Each tray's HSCs latch an input under-voltage flag when the tray's 12 V input sags, and an over-current fault when they
trip on the tray's current (usually turning the tray off). A weak bus bar usually does not fail the test that sags it:
the chip survives and the flag stays latched. So the flags are read before and after the load and a newly set one is
reported.

Every transaction is one ipmitool call under one box-wide lock file, so two readers never overlap on the BMC's I2C
master (overlapping write-reads garble the command byte and latch the HSCs' communication-fault bit). Only reads,
except two opt-in writes: CLEAR_FAULTS to a tray HSC, and the brick mux's channel select and its restore.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager

from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# Tray number -> I2C bus of its HSCs, and the two HSC addresses on each.
HSC_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
HSC_ADDRS = (0x40, 0x42)
TRAY_CPLD = 0x80  # on the same bus as the tray's HSCs

# PMBus registers read: name -> (code, bytes). The two peaks clear when read.
REG = {
    "STATUS_WORD": (0x79, 2),
    "STATUS_VOUT": (0x7A, 1),
    "STATUS_IOUT": (0x7B, 1),
    "STATUS_INPUT": (0x7C, 1),
    "STATUS_TEMPERATURE": (0x7D, 1),
    "STATUS_CML": (0x7E, 1),
    "STATUS_MFR_SPECIFIC": (0x80, 1),
    "READ_VIN": (0x88, 2),
    "READ_VOUT": (0x8B, 2),
    "READ_IOUT": (0x8C, 2),
    "READ_PIN": (0x97, 2),
    "FAULT_RECORD": (0xFA, 2),
    "FAULT_RECORD_LAST": (0xFB, 2),
    "IOUT_OC_FAULT_LIMIT": (0x46, 2),
    "IOUT_OC_WARN_LIMIT": (0x4A, 2),
    "VIN_OV_FAULT_LIMIT": (0x55, 2),
    "VIN_OV_WARN_LIMIT": (0x57, 2),
    "VIN_UV_WARN_LIMIT": (0x58, 2),
    "VIN_UV_FAULT_LIMIT": (0x59, 2),
    "READ_IOUT_PEAK": (0xA6, 2),
    "READ_TEMP_PEAK": (0xAF, 2),
}
STATUS_WORD, STATUS_INPUT = REG["STATUS_WORD"], REG["STATUS_INPUT"]
VIN_LIMITS = ("VIN_OV_FAULT_LIMIT", "VIN_OV_WARN_LIMIT", "VIN_UV_WARN_LIMIT", "VIN_UV_FAULT_LIMIT")
VIN_LSB, IOUT_LSB = 0.03125, 0.0625  # V and A per count (READ_VIN, READ_VOUT; READ_IOUT, READ_IOUT_PEAK)
CLEAR_FAULTS = 0x03

# STATUS_WORD bits (an HSC that turned itself off, an over-current trip) and
# STATUS_INPUT bits, in the order a reading names them.
WORD_BITS = {0x40: "OFF", 0x10: "IOUT_OC_FAULT"}
INPUT_BITS = {0x10: "VIN_UV_FAULT", 0x20: "VIN_UV_WARN"}
FLAG_ORDER = (*WORD_BITS.values(), *INPUT_BITS.values())
CML = 0x02  # STATUS_WORD's communication-fault bit: not power

# BMC sensors (Get Sensor Reading): number -> scale. Power_UBB0-3 (trays 1-4, W), the HSC sum (W), Power_PSU0-3 (W),
# inlet and outlet air (C).
SENSORS = {
    0x35: 10,
    0x36: 10,
    0x37: 10,
    0x38: 10,
    0x60: 2,
    0x31: 20,
    0x32: 20,
    0x33: 20,
    0x34: 20,
    0x1E: 1,
    0x1F: 1,
    0x3E: 1,
}

# The power bricks U1-U8 (two per mux channel) and the 54 V fan rail's current monitor, on the box's power board.
BRICK_BUS, BRICK_MUX = 0x09, 0xEC
BRICKS = {f"U{n}": ((n - 1) // 2, 0xC0 if n % 2 else 0xC2) for n in range(1, 9)}  # name -> (mux channel, address)
FAN_BUS, FAN_ADDR, FAN_CUR_LSB = 0x01, 0x80, 43.04 / 32768

READ_TIMEOUT_S = 5
LOCK_WAIT_S = 10

HscStatus = dict[tuple[int, int], "tuple[int, int] | None"]  # (tray, addr) -> (STATUS_WORD, STATUS_INPUT)


def lin11(x: int) -> float:
    """A PMBus LINEAR11 word."""
    e, m = x >> 11, x & 0x7FF
    return (m - 2048 if m >= 1024 else m) * 2.0 ** (e - 32 if e >= 16 else e)


def tray_bdfs(tray: int) -> str:
    """'81:00.0-88:00.0': the tray's chips."""
    n = {1: 0, 2: 4, 3: 0xC, 4: 8}[tray]
    return f"{n:x}1:00.0-{n:x}8:00.0"


class Bmc:
    """The in-band BMC route: ipmitool as root, else ``sudo -n ipmitool`` when ``sudo -n -l`` lists the exact read.
    ``why`` says why there is no route (``cmd`` is then empty)."""

    def __init__(self, ipmitool: str | None = None, lock: str | None = None):
        self.lock = lock or f"/tmp/hc-bmc-{os.getuid()}.lock"
        self.cmd: list[str] = []
        self.why = ""
        self._listed: dict[tuple, bool] = {}
        if ipmitool:
            self.cmd = shlex.split(ipmitool)
            self.why = "" if shutil.which(self.cmd[0]) else f"{self.cmd[0]} not found"
        elif not shutil.which("ipmitool"):
            self.why = "ipmitool not found"
        elif os.geteuid() == 0:
            self.cmd = ["ipmitool"]
        else:
            self.cmd = ["sudo", "-n", "ipmitool"]
            if not self.listed("raw", "0x06", "0x52", "0x11", "0x40", "0x02", "0x79"):
                self.why = "no root, and sudo -n -l does not list ipmitool raw 0x06 0x52"
        if self.why:
            self.cmd = []

    def listed(self, *args: str) -> bool:
        """Whether this exact ipmitool command may run: always as root or a given command; under sudo only when
        ``sudo -n -l`` lists it (some sites list fixed command lines only)."""
        if self.cmd[:1] != ["sudo"]:
            return True
        if args not in self._listed:
            self._listed[args] = sudo_listed(["ipmitool", *args])
        return self._listed[args]

    @contextmanager
    def locked(self):
        """The box-wide I2C lock, held for one transaction (or one mux select-read-restore). A holder past the wait, or a
        lock file this user cannot open, does not stop the read."""
        try:
            f = open(self.lock, "a")
        except OSError:
            yield
            return
        with f:
            for _ in range(LOCK_WAIT_S * 10):
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def raw(self, *args: int, lock: bool = True) -> list[int] | None:
        """One ``ipmitool raw`` call; the bytes it returned, None when it failed or there is no route."""
        if not self.cmd:
            return None
        cmd = [*self.cmd, "raw", *(f"{a:#04x}" for a in args)]
        try:
            if lock:
                with self.locked():
                    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=READ_TIMEOUT_S)
            else:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=READ_TIMEOUT_S)
            return [int(b, 16) for b in cp.stdout.split()] if cp.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None

    def i2c(self, bus: int, addr: int, code: int, nbytes: int, lock: bool = True) -> int | None:
        """A PMBus/SMBus read (master write-read), little-endian; None when it did not answer in full."""
        b = self.raw(0x06, 0x52, bus, addr, nbytes, code, lock=lock)
        return int.from_bytes(bytes(b), "little") if b is not None and len(b) == nbytes else None

    def hsc(self, tray: int, addr: int, name: str) -> int | None:
        code, n = REG[name]
        return self.i2c(HSC_BUS[tray], addr, code, n)

    def clear_faults(self, tray: int, addr: int) -> bool:
        """PMBus CLEAR_FAULTS (a send-byte): resets the HSC's status latches and nothing else."""
        return self.raw(0x06, 0x52, HSC_BUS[tray], addr, 0x00, CLEAR_FAULTS) is not None

    def sensor(self, num: int) -> str:
        """One BMC sensor scaled, 'NA' when the BMC marks it unavailable, 'ERR' when it did not answer."""
        b = self.raw(0x04, 0x2D, num)
        if not b or len(b) < 2:
            return "ERR"
        return "NA" if b[1] & 0x20 else str(b[0] * SENSORS[num])

    def cpld(self, tray: int, reg: int) -> int | None:
        """One byte of a tray CPLD: 0x19 VR power-good, 0x1E chip alive, 0x2D thermal trip (one bit per chip)."""
        return self.i2c(HSC_BUS[tray], TRAY_CPLD, reg, 1)


def sudo_listed(argv: list[str]) -> bool:
    try:
        return subprocess.run(["sudo", "-n", "-l", *argv], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_regs(bmc: Bmc, names, trays=HSC_BUS) -> dict[tuple[int, int], dict[str, int | None]]:
    """The named registers of every HSC of *trays*; an HSC that does not answer the first is not read further (each
    miss costs a timeout). Under sudo a VIN limit is read only where its exact command is listed."""
    out = {}
    for t in trays:
        for a in HSC_ADDRS:
            r = out[(t, a)] = {}
            for n in names:
                code, nb = REG[n]
                listed = n not in VIN_LIMITS or bmc.listed(
                    "raw", *(f"{x:#04x}" for x in (6, 0x52, HSC_BUS[t], a, nb, code))
                )
                r[n] = bmc.hsc(t, a, n) if listed and (n == names[0] or r[names[0]] is not None) else None
    return out


def read_hsc_status(bmc: Bmc | None = None) -> HscStatus:
    """STATUS_WORD and STATUS_INPUT of every tray HSC; None for one that did not answer."""
    regs = read_regs(bmc or Bmc(), ("STATUS_WORD", "STATUS_INPUT"))
    return {k: None if None in r.values() else (r["STATUS_WORD"], r["STATUS_INPUT"]) for k, r in regs.items()}


def _flags(reading: "tuple[int, int]") -> set[str]:
    word, inp = reading
    return {name for bit, name in WORD_BITS.items() if word & bit} | {
        name for bit, name in INPUT_BITS.items() if inp & bit
    }


def flag_names(flags: set[str]) -> str:
    """The flags as one reason, e.g. 'OFF, IOUT_OC_FAULT' for an over-current trip."""
    return ", ".join(f for f in FLAG_ORDER if f in flags)


def vin_limit(name: str, raw: int | None) -> str:
    """A VIN limit as text: bits 6:0 x 125 mV; a UV fault limit of 0 means UV fault reporting is off."""
    if raw is None:
        return "not read"
    if name == "VIN_UV_FAULT_LIMIT" and raw % 128 == 0:
        return "0 V (UV fault reporting disabled)"
    return f"{(raw % 128) * 0.125:.2f} V"


def _brick(bmc: Bmc, addr: int) -> tuple:
    """(volts, amps, watts, temp_C, vin) of one brick, None = not read. Vout uses the brick's own VOUT_MODE exponent:
    bricks of one box differ (2^-10 or 2^-12 V), so a fixed scale reads some 4x off."""
    rd = lambda code, n=2: bmc.i2c(BRICK_BUS, addr, code, n, lock=False)  # noqa: E731
    mode, vout, vin, iout, temp = rd(0x20, 1), rd(0x8B), rd(0x88), rd(0x8C), rd(0x8D)
    v = vout * 2.0 ** ((mode & 31) - (32 if mode & 16 else 0)) if None not in (mode, vout) and mode >> 5 == 0 else None
    i = lin11(iout) if iout is not None else None
    lin = lambda x: lin11(x) if x is not None else None  # noqa: E731
    return v, i, v * i if None not in (v, i) else None, lin(temp), lin(vin)


def brick_pass(bmc: Bmc) -> tuple[list[tuple], list[str]]:
    """One pass over the bricks and the fan rail: ``(name, volts, amps, watts, temp_C, vin)`` rows and notes. Per mux
    channel, under one hold of the I2C lock: read the mux's control byte, select the channel, read its two bricks and
    write the saved byte back, also on an error."""
    rows, notes = [], []
    for ch in range(4):
        with bmc.locked():
            saved = bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x01, lock=False)
            saved = saved[0] if saved and len(saved) == 1 else None
            ok = saved is not None and bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, 1 << ch, lock=False) is not None
            try:
                rows += [(n, *(_brick(bmc, a) if ok else (None,) * 5)) for n, (c, a) in BRICKS.items() if c == ch]
            finally:
                if saved is not None and bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, saved, lock=False) is None:
                    notes.append(
                        f"brick mux restore failed: bus {BRICK_BUS:#04x} mux {BRICK_MUX:#04x} (was {saved:#04x})"
                    )

    def be(code: int, n: int) -> int | None:  # the fan rail monitor's registers are big-endian
        b = bmc.raw(0x06, 0x52, FAN_BUS, FAN_ADDR, n, code)
        return int.from_bytes(bytes(b), "big") if b and len(b) == n else None

    vbus, cur, pw = be(0x05, 2), None, None
    if be(0x02, 2):  # SHUNT_CAL is only read: unset, the current and power read nothing
        c, p = be(0x07, 2), be(0x08, 3)
        cur = (c - 65536 if c >= 32768 else c) * FAN_CUR_LSB if c is not None else None
        pw = p * 0.2 * FAN_CUR_LSB if p is not None else None
    else:
        notes.append("fan rail current not read: SHUNT_CAL unset")
    rows.append(("FAN", vbus * 0.003125 if vbus is not None else None, cur, pw, None, None))
    return rows, notes


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
