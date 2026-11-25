# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tray power reads through the in-band BMC (``ipmitool raw``): the tray hot-swap controllers (HSCs), the tray CPLDs'
power-good byte, the BMC's power sensors and, opt-in, the power bricks. A weak bus bar seldom fails the test that sags
it, but the HSC's under-voltage flag stays latched, so the flags are read before and after the load. Every transaction
takes one box-wide lock (overlapping I2C write-reads latch the HSCs' comm-fault bit). Only reads, except the opt-in
CLEAR_FAULTS and brick mux select/restore."""
from __future__ import annotations

import fcntl
import os
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager, nullcontext

from .common import root_cmd
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# Tray number -> I2C bus of its HSCs, and the two HSC addresses on each.
HSC_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
HSC_ADDRS = (0x40, 0x42)
TRAY_CPLD = 0x80  # on the same bus as the tray's HSCs

# PMBus registers: name -> (code, bytes). The two peaks clear when read.
_R = """STATUS_WORD 79 2 STATUS_VOUT 7A 1 STATUS_IOUT 7B 1 STATUS_INPUT 7C 1 STATUS_TEMPERATURE 7D 1 STATUS_CML 7E 1
STATUS_MFR_SPECIFIC 80 1 READ_VIN 88 2 READ_VOUT 8B 2 READ_IOUT 8C 2 READ_PIN 97 2 FAULT_RECORD FA 2
FAULT_RECORD_LAST FB 2 IOUT_OC_FAULT_LIMIT 46 2 IOUT_OC_WARN_LIMIT 4A 2 VIN_OV_FAULT_LIMIT 55 2 VIN_OV_WARN_LIMIT 57 2
VIN_UV_WARN_LIMIT 58 2 VIN_UV_FAULT_LIMIT 59 2 READ_IOUT_PEAK A6 2 READ_TEMP_PEAK AF 2""".split()
REG = {n: (int(c, 16), int(b)) for n, c, b in zip(_R[::3], _R[1::3], _R[2::3])}
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

# BMC sensor number -> scale: Power_UBB0-3 (trays 1-4), an HSC power sensor, Power_PSU0-3 (W); inlet and outlet air (C).
SENSORS = dict(zip((0x35, 0x36, 0x37, 0x38, 0x60), (10, 10, 10, 10, 2)))
SENSORS |= dict(zip((0x31, 0x32, 0x33, 0x34, 0x1E, 0x1F, 0x3E), (20, 20, 20, 20, 1, 1, 1)))

# The power bricks U1-U8 (two per mux channel) and the 54 V fan rail's current monitor, on the box's power board.
BRICK_BUS, BRICK_MUX = 0x09, 0xEC
BRICKS = {f"U{n}": ((n - 1) // 2, 0xC0 if n % 2 else 0xC2) for n in range(1, 9)}  # name -> (mux channel, address)
FAN_BUS, FAN_ADDR, FAN_CUR_LSB = 0x01, 0x80, 43.04 / 32768

READ_TIMEOUT_S = 5
LOCK_WAIT_S = 10

HscStatus = dict[tuple[int, int], "tuple[int, int] | None"]  # (tray, addr) -> (STATUS_WORD, STATUS_INPUT)


def lin11(x: int) -> float:  # a PMBus LINEAR11 word
    e, m = x >> 11, x & 0x7FF
    return (m - 2048 if m >= 1024 else m) * 2.0 ** (e - 32 if e >= 16 else e)


def tray_bdfs(tray: int) -> str:  # '81:00.0-88:00.0': the tray's chips
    n = {1: 0, 2: 4, 3: 0xC, 4: 8}[tray]
    return f"{n:x}1:00.0-{n:x}8:00.0"


class Bmc:
    """The in-band BMC route: ipmitool as root, else ``sudo -n ipmitool`` when ``sudo -n -l`` lists the exact read.
    ``why`` says why there is no route (``cmd`` is then empty)."""

    def __init__(self, ipmitool: str | None = None, lock: str | None = None):
        self.lock, self.cmd, self.why, self._listed = lock or f"/tmp/hc-bmc-{os.getuid()}.lock", [], "", {}
        self.failed = ""  # why the last raw call failed: 'ipmitool rc N' (124: it timed out), or the OSError class
        if ipmitool:
            self.cmd = shlex.split(ipmitool)
            self.why = "" if shutil.which(self.cmd[0]) else f"{self.cmd[0]} not found"
        elif not shutil.which("ipmitool"):
            self.why = "ipmitool not found"
        else:
            self.cmd = (root_cmd(["ipmitool", "raw", "0x06", "0x52", "0x11", "0x40", "0x02", "0x79"])[0] or [])[:-7]
            self.why = "" if self.cmd else "no root, and sudo -n -l does not list ipmitool raw 0x06 0x52"
        if self.why:
            self.cmd = []

    def listed(self, *args: str) -> bool:
        """Whether this exact ipmitool command may run: under sudo only when ``sudo -n -l`` lists it."""
        if self.cmd[:1] == ["sudo"] and args not in self._listed:
            self._listed[args] = root_cmd(["ipmitool", *args])[0] is not None
        return self._listed.get(args, True)

    @contextmanager
    def locked(self):
        """The box-wide I2C lock; a holder past LOCK_WAIT_S or an unopenable lock file does not stop the read."""
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
        self.failed = "no route"
        if not self.cmd:
            return None
        try:
            with self.locked() if lock else nullcontext():
                cp = subprocess.run(
                    [*self.cmd, "raw", *(f"{a:#04x}" for a in args)],
                    capture_output=True,
                    text=True,
                    timeout=READ_TIMEOUT_S,
                )
            self.failed = f"ipmitool rc {cp.returncode}"
            if cp.returncode:
                return None
            self.failed = f"not hex: {cp.stdout.strip()[:16]}"
            b = [int(x, 16) for x in cp.stdout.split()]
            self.failed = ""
            return b
        except subprocess.TimeoutExpired:
            self.failed = "ipmitool rc 124"
        except OSError as e:
            self.failed = type(e).__name__
        except ValueError:
            pass
        return None

    def i2c(self, bus: int, addr: int, code: int, nbytes: int, lock: bool = True) -> int | None:
        """A PMBus/SMBus read (master write-read), little-endian; None when it did not answer in full."""
        b = self.raw(0x06, 0x52, bus, addr, nbytes, code, lock=lock)
        return int.from_bytes(bytes(b), "little") if b is not None and len(b) == nbytes else None

    def block(self, bus: int, addr: int, code: int, lock: bool = True) -> bytes | None:
        """A PMBus block read: the length byte, then that many bytes; None when it did not answer or sent none."""
        b = self.raw(0x06, 0x52, bus, addr, 33, code, lock=lock)
        return bytes(b[1 : 1 + b[0]]) if b and 0 < b[0] < len(b) else None

    def hsc(self, tray: int, addr: int, name: str) -> int | None:
        return self.i2c(HSC_BUS[tray], addr, *REG[name])

    def clear_faults(self, tray: int, addr: int) -> bool:  # a send-byte: resets the HSC's status latches only
        return self.raw(0x06, 0x52, HSC_BUS[tray], addr, 0x00, CLEAR_FAULTS) is not None

    def sensor(self, num: int) -> str:
        """One BMC sensor scaled, 'NA' when the BMC marks it unavailable, 'ERR' when it did not answer."""
        b = self.raw(0x04, 0x2D, num)
        return "ERR" if not b or len(b) < 2 else "NA" if b[1] & 0x20 else str(b[0] * SENSORS[num])

    def cpld(self, tray: int, reg: int) -> int | None:  # 0x19 VR power-good, 0x1E chip alive, 0x2D thermal trip
        return self.i2c(HSC_BUS[tray], TRAY_CPLD, reg, 1)


def read_regs(bmc: Bmc, names, trays=HSC_BUS) -> dict[tuple[int, int], dict[str, int | None]]:
    """The named registers of every HSC of *trays*; an HSC that does not answer the first is not read further (each
    miss costs a timeout). Under sudo a VIN limit is read only where its exact command is listed."""
    out = {}
    for t in trays:
        for a in HSC_ADDRS:
            r = out[(t, a)] = {}
            for n in names:
                code, nb = REG[n]
                cmd = ("raw", *(f"{x:#04x}" for x in (6, 0x52, HSC_BUS[t], a, nb, code)))
                listed = n not in VIN_LIMITS or bmc.listed(*cmd)
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
    """A VIN limit: bits 6:0 x 125 mV; a UV fault limit of 0 turns UV fault reporting off."""
    if raw is None:
        return "not read"
    return (
        "0 V (UV fault reporting disabled)"
        if name == "VIN_UV_FAULT_LIMIT" and raw % 128 == 0
        else f"{(raw % 128) * 0.125:.2f} V"
    )


BRICK_ID = (("MFR_ID", 0x99), ("MFR_MODEL", 0x9A), ("MFR_REVISION", 0x9B), ("MFR_DATE", 0x9D), ("MFR_SERIAL", 0x9E))


def _exp(mode: int) -> int:  # the exponent of a VOUT_MODE byte: 5 bits, two's complement
    return (mode & 31) - (32 if mode & 16 else 0)


def _brick(bmc: Bmc, name: str, addr: int, modes: dict, load: bool, stop, fails: list, ident: dict | None) -> tuple:
    """(volts, amps, watts, temp_C, vin) of one brick, None = not read. Vout by the brick's own VOUT_MODE exponent:
    bricks of one box differ (2^-10 or 2^-12 V); *modes* caches it per brick. Under *load* only VOUT and VIN, and
    VOUT_MODE only while not cached. A read that fails, other than by timeout, is read once more (the BMC's own I2C traffic drops
    single reads), then named in *fails*; once *stop* says so nothing more is read. *ident* (the idle pass): the brick's
    MFR_* strings and VOUT_MODE and VOUT_COMMAND go in it as text, None = not read."""

    def rd(code: int, n: int = 0) -> int | bytes | None:  # n = 0: a block read
        for _ in range(2):
            if stop():
                return None
            x = bmc.i2c(BRICK_BUS, addr, code, n, False) if n else bmc.block(BRICK_BUS, addr, code, False)
            if x is not None:
                return x
            if bmc.failed == "ipmitool rc 124":  # a timeout is not read again: it would hold the I2C lock twice as long
                break
        why = bmc.failed or "short answer"
        fails.append(f"brick read failed: {name} bus {BRICK_BUS:#04x} addr {addr:#04x} code {code:#04x} ({why})")
        return None

    if (mode := modes.get(name) if load else None) is None and (mode := rd(0x20, 1)) is not None:
        modes[name] = mode
    if ident is not None:
        ident[name] = {
            k: None if (x := rd(c)) is None else x.decode("ascii", "replace").strip(" \x00") for k, c in BRICK_ID
        }
        ident[name]["VOUT_MODE"] = None if mode is None else f"{mode:#04x}"
        cmd = rd(0x21, 2)
        ident[name]["VOUT_COMMAND"] = None if None in (mode, cmd) or mode >> 5 else f"{cmd * 2.0 ** _exp(mode):.4f} V"
    vout, vin = rd(0x8B, 2), rd(0x88, 2)
    iout, temp = (None, None) if load else (rd(0x8C, 2), rd(0x8D, 2))
    v = vout * 2.0 ** _exp(mode) if None not in (mode, vout) and mode >> 5 == 0 else None
    i, temp, vin = (None if x is None else lin11(x) for x in (iout, temp, vin))
    return v, i, v * i if None not in (v, i) else None, temp, vin


def brick_pass(
    bmc: Bmc, modes: dict | None = None, load: bool = False, stop=lambda: False, ident: dict | None = None
) -> tuple[list[tuple], list[str], list[str]]:
    """One pass over the bricks and the fan rail: ``(name, volts, amps, watts, temp_C, vin)`` rows, notes and read
    failures. Per mux channel, under one hold of the I2C lock: save the mux's control byte, select the channel, read
    its two bricks and write the saved byte back, also on an error. *load*: the pass under load reads only each
    brick's VOUT and VIN (VOUT_MODE from *modes*) and the fan rail's VBUS. Once *stop* says so it reads nothing
    more: the mux is still restored and the rows so far returned. *ident*: the idle pass also reads each brick's
    MFR_* strings, VOUT_MODE and VOUT_COMMAND into it (name -> {field: text})."""
    rows, notes, fails, modes = [], [], [], {} if modes is None else modes
    for ch in range(4):
        if stop():
            return rows, notes, fails
        with bmc.locked():
            if stop():  # a stop during the lock wait: no mux write left to restore
                return rows, notes, fails
            saved = bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x01, lock=False)
            saved = saved[0] if saved and len(saved) == 1 else None
            ok = saved is not None and bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, 1 << ch, lock=False) is not None
            if not ok:
                fails.append(f"brick mux read or select failed: bus {BRICK_BUS:#04x} mux {BRICK_MUX:#04x} channel {ch}")
            try:
                rows += [
                    (n, *(_brick(bmc, n, a, modes, load, stop, fails, ident) if ok else (None,) * 5))
                    for n, (c, a) in BRICKS.items()
                    if c == ch
                ]
            finally:
                if saved is not None and bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, saved, lock=False) is None:
                    notes.append(
                        f"brick mux restore failed: bus {BRICK_BUS:#04x} mux {BRICK_MUX:#04x} (was {saved:#04x})"
                    )

    def be(code: int, n: int) -> int | None:  # the fan rail monitor's registers are big-endian
        b = bmc.raw(0x06, 0x52, FAN_BUS, FAN_ADDR, n, code)
        return int.from_bytes(bytes(b), "big") if b and len(b) == n else None

    if stop():
        return rows, notes, fails
    vbus, cur, pw = be(0x05, 2), None, None
    if vbus is None:
        fails.append(f"54 V rail read failed: bus {FAN_BUS:#04x} addr {FAN_ADDR:#04x} code 0x05")
    if load:  # under load the rail's VBUS only
        pass
    elif be(0x02, 2):  # SHUNT_CAL is only read: unset, the current and power read nothing
        c, p = be(0x07, 2), be(0x08, 3)
        cur = (c - 65536 if c >= 32768 else c) * FAN_CUR_LSB if c is not None else None
        pw = p * 0.2 * FAN_CUR_LSB if p is not None else None
    else:
        notes.append("fan rail current not read: SHUNT_CAL unset")
    rows.append(("FAN", vbus * 0.003125 if vbus is not None else None, cur, pw, None, None))
    return rows, notes, fails


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
