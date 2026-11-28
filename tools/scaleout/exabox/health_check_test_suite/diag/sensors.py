# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tray power through the in-band BMC (``ipmitool raw 0x06 0x52``): each tray's two
hot-swap controllers (HSCs), its CPLD's VR power-good byte and, where the power
board has them, the power bricks.

A weak bus bar seldom fails the test that sags it, but the HSC latches an input
under-voltage flag, so the flags are read before and after the load, judged,
then cleared once per HSC (PMBus CLEAR_FAULTS) so the next run starts clean.
The same reads tell a lost chip's cause: its tray's 12 V cut (POWER), or 12 V
good with a VR's power-good low (POWER-DOWNSTREAM).

Every ipmitool call takes one box-wide lock, ``$HSC_I2C_LOCK`` (the lock
tray-test uses too): overlapping I2C write-reads latch the HSCs' comm-fault bit.
The only writes are CLEAR_FAULTS and the brick mux's channel select, written
back after. Needs root or ``sudo -n ipmitool``; without either the reads are one
'skipped' line.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from .common import log
from .galaxy import expected_bdf, format_bdf
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

TRAY_POWER = "tray_power"  # check names this module emits

# Tray -> I2C bus of its HSCs (and its CPLD), and the two HSC addresses on each.
HSC_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
HSC_ADDRS = (0x40, 0x42)
TRAY_CPLD, CPLD_VR_PG = 0x80, 0x19  # VR power-good byte: 0xFF when every VR's is high
VR_PG_ALL = 0xFF

# PMBus registers: (code, bytes).
STATUS_WORD, STATUS_INPUT, READ_VIN, READ_IOUT = (0x79, 2), (0x7C, 1), (0x88, 2), (0x8C, 2)
VOUT_MODE, READ_VOUT = (0x20, 1), (0x8B, 2)
VIN_LSB, IOUT_LSB = 0.03125, 0.0625  # V and A per count of the HSCs' READ_VIN, READ_IOUT
CLEAR_FAULTS = 0x03

# STATUS_WORD bits (the HSC turned the tray's 12 V off; an over-current trip),
# then STATUS_INPUT's under-voltage latches, in the order a reading names them.
WORD_BITS = {0x40: "OFF", 0x10: "IOUT_OC_FAULT"}
INPUT_BITS = {0x10: "VIN_UV_FAULT", 0x20: "VIN_UV_WARN"}
TRIPS = frozenset(WORD_BITS.values())
VIN_LOW_V = 11.0  # under this, the tray's 12 V is not good, latch or not

# The power bricks U1-U8, two per channel of the mux on the power board's bus.
BRICK_BUS, BRICK_MUX = 0x09, 0xEC
BRICKS = {f"U{n}": ((n - 1) // 2, 0xC0 if n % 2 else 0xC2) for n in range(1, 9)}

READ_TIMEOUT_S = 5
LOCK_WAIT_S = 10


def lin11(x: int) -> float:
    """A PMBus LINEAR11 word."""
    e, m = x >> 11, x & 0x7FF
    return (m - 2048 if m >= 1024 else m) * 2.0 ** (e - 32 if e >= 16 else e)


@dataclass
class Hsc:
    word: int
    inp: int
    vin: float | None = None
    iout: float | None = None

    @property
    def flags(self) -> list[str]:
        return [n for b, n in WORD_BITS.items() if self.word & b] + [n for b, n in INPUT_BITS.items() if self.inp & b]

    def __str__(self) -> str:
        vin = "VIN not read" if self.vin is None else f"{self.vin:.2f} V"
        iout = "" if self.iout is None else f" {self.iout:.1f} A"
        return f"{vin}{iout} {', '.join(self.flags) or 'no flags'}"


@dataclass
class TrayPower:
    """One pass over the trays: (tray, HSC address) -> reading, None when it did not
    answer; tray -> VR power-good byte; brick -> (VOUT V, IOUT A)."""

    hsc: dict[tuple[int, int], Hsc | None] = field(default_factory=dict)
    vr_pg: dict[int, int | None] = field(default_factory=dict)
    bricks: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)


class Bmc:
    """The in-band BMC route: ipmitool as root, else under ``sudo -n``. ``why`` says
    why there is no route; once set, nothing more is sent."""

    def __init__(self, dry_run: bool = False):
        self.lock = os.environ.get("HSC_I2C_LOCK") or f"/tmp/tray-bmc-{os.getuid()}.lock"
        self.cmd = ["ipmitool"] if os.geteuid() == 0 else ["sudo", "-n", "ipmitool"]
        self.why = "dry run" if dry_run else "" if shutil.which("ipmitool") else "ipmitool not found"

    @contextmanager
    def locked(self, wait: float | None = None, who: str = "health check", fatal: bool = True):
        """Hold the box-wide I2C lock, writing ``who`` into it so a waiting poll can name the holder.
        Yields False when another holder kept it past ``wait`` (``fatal`` then ends the route) or
        the lock file cannot be opened (that always ends the route)."""
        wait = LOCK_WAIT_S if wait is None else wait
        try:
            f = open(self.lock, "a+")
        except OSError as e:
            self.why = f"cannot open the I2C lock {self.lock} ({e.strerror})"
            yield False
            return
        with f:
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        if fatal:
                            self.why = f"I2C lock busy after {wait:g} s"
                        yield False
                        return
                    time.sleep(0.005)
            try:
                f.truncate(0)
                f.write(f"{who} pid {os.getpid()}\n")
                f.flush()
                yield True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def holder(self) -> str:
        """Who last took the lock, as it wrote itself in; "unknown" for a reader that does not."""
        try:
            return Path(self.lock).read_text().strip() or "unknown"
        except OSError:
            return "unknown"

    def raw(self, *args: int, lock: bool = True) -> list[int] | None:
        """One ``ipmitool raw`` call, under the lock unless the caller holds it: the bytes
        returned, None when it failed. A sudo refusal, a missing sudo or BMC device ends the route."""
        if self.why:
            return None
        cmd = [*self.cmd, "raw", *(f"{a:#04x}" for a in args)]
        try:
            with self.locked() if lock else nullcontext(True) as held:
                if not held:
                    return None
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=READ_TIMEOUT_S)
        except OSError as e:
            self.why = f"cannot run {cmd[0]} ({e.strerror})"
            return None
        except subprocess.TimeoutExpired:
            return None
        if cp.returncode:
            err = (cp.stderr or "").strip().splitlines()
            if err and err[0].startswith("sudo:"):
                self.why = f"no root, and sudo -n ipmitool was refused ({err[0]})"
            elif err and err[0].startswith("Could not open device"):
                self.why = f"no BMC device ({err[0]})"
            return None
        try:
            return [int(x, 16) for x in cp.stdout.split()]
        except ValueError:
            return None

    def read(self, bus: int, addr: int, reg: tuple[int, int], lock: bool = True) -> int | None:
        """An I2C register read (master write-read), little-endian; None unless it answered in full."""
        code, n = reg
        b = self.raw(0x06, 0x52, bus, addr, n, code, lock=lock)
        return int.from_bytes(bytes(b), "little") if b is not None and len(b) == n else None

    def clear_faults(self, tray: int, addr: int) -> bool:
        """PMBus CLEAR_FAULTS (a send-byte): resets the HSC's status latches only."""
        return self.raw(0x06, 0x52, HSC_BUS[tray], addr, 0x00, CLEAR_FAULTS) is not None


def read_hsc(bmc: Bmc, tray: int, addr: int, lock: bool = True) -> Hsc | None:
    """One HSC's status and input; None when STATUS_WORD or STATUS_INPUT did not answer."""
    bus = HSC_BUS[tray]
    word = bmc.read(bus, addr, STATUS_WORD, lock)
    inp = None if word is None else bmc.read(bus, addr, STATUS_INPUT, lock)
    if inp is None:
        return None
    vin, iout = bmc.read(bus, addr, READ_VIN, lock), bmc.read(bus, addr, READ_IOUT, lock)
    return Hsc(word, inp, None if vin is None else vin * VIN_LSB, None if iout is None else iout * IOUT_LSB)


def read_bricks(bmc: Bmc) -> dict[str, tuple[float | None, float | None]]:
    """Each brick's VOUT (by its own VOUT_MODE exponent) and IOUT; empty when the box
    has no brick mux. Per mux channel, under one hold of the lock: save the mux's
    control byte, select the channel, read its two bricks, write the byte back."""
    out = {}
    for ch in range(4):
        with bmc.locked() as held:
            saved = bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x01, lock=False) if held else None
            if not saved or len(saved) != 1:
                return out  # no mux answering: no bricks on this box
            try:
                if bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, 1 << ch, lock=False) is None:
                    continue
                for name, (c, addr) in BRICKS.items():
                    if c != ch:
                        continue
                    mode, vout, iout = (
                        bmc.read(BRICK_BUS, addr, r, lock=False) for r in (VOUT_MODE, READ_VOUT, READ_IOUT)
                    )
                    exp = None if mode is None or mode >> 5 else (mode & 31) - (32 if mode & 16 else 0)
                    out[name] = (
                        None if None in (exp, vout) else vout * 2.0**exp,
                        None if iout is None else lin11(iout),
                    )
            finally:
                if bmc.raw(0x06, 0x52, BRICK_BUS, BRICK_MUX, 0x00, saved[0], lock=False) is None:
                    log(f"tray power: brick mux restore failed (was {saved[0]:#04x})")
    return out


def read_power(bmc: Bmc, bricks: bool = True) -> TrayPower | None:
    """Every tray's HSCs and VR power-good, and the bricks unless told not to; None when there is no route."""
    if bmc.why:
        return None
    tp = TrayPower()
    for t, bus in HSC_BUS.items():
        for a in HSC_ADDRS:
            tp.hsc[(t, a)] = read_hsc(bmc, t, a)
        tp.vr_pg[t] = bmc.read(bus, TRAY_CPLD, (CPLD_VR_PG, 1))
    tp.bricks = read_bricks(bmc) if bricks else {}
    return None if bmc.why else tp


POLL_S = 2.0


@dataclass
class PollStep:
    """What the HSC poll saw during one bus-bar step, per (tray, HSC address)."""

    peak_a: dict[tuple[int, int], float] = field(default_factory=dict)
    vin_min: dict[tuple[int, int], float] = field(default_factory=dict)
    flags: dict[tuple[int, int], set[str]] = field(default_factory=dict)


@dataclass
class Poll:
    """The poll's record: the step the caller is in, what each step saw, and the read gaps."""

    step: str = "start"
    steps: dict[str, PollStep] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


def poll_hsc(bmc: Bmc, poll: Poll, stop: threading.Event, period: float = POLL_S) -> None:
    """Every ``period`` s until ``stop``: each HSC under its own short hold of the lock, into
    ``poll.steps[poll.step]``. A hold not got within a period misses that read; a gap over two
    periods between reads is logged with the lock's holder."""
    last, blocker = time.monotonic(), ""
    while not stop.is_set() and not bmc.why:
        rec = poll.steps.setdefault(poll.step, PollStep())
        for key in ((t, a) for t in HSC_BUS for a in HSC_ADDRS):
            with bmc.locked(wait=period, who="health check poll", fatal=False) as held:
                h = read_hsc(bmc, *key, lock=False) if held else None
            if not held:
                blocker = bmc.holder()
            if h is None:
                continue
            now = time.monotonic()
            if now - last > 2 * period:
                poll.gaps.append(
                    f"HSC poll gap {now - last:.1f} s in {poll.step} (lock held by {blocker or 'unknown'})"
                )
                log(poll.gaps[-1])
            last, blocker = now, ""
            rec.flags.setdefault(key, set()).update(h.flags)
            if h.iout is not None:
                rec.peak_a[key] = max(rec.peak_a.get(key, 0.0), h.iout)
            if h.vin is not None:
                rec.vin_min[key] = min(rec.vin_min.get(key, h.vin), h.vin)
        stop.wait(period)


def _tray_parts(tray: int) -> list[str]:
    return [format_bdf(expected_bdf(tray, asic)) for asic in range(1, 9)]


def _tripped(h: Hsc | None) -> bool:
    """OFF, over-current or VIN under VIN_LOW_V: the tray lost its 12 V, not just sagged."""
    return bool(h and (TRIPS & set(h.flags) or (h.vin is not None and h.vin < VIN_LOW_V)))


class TrayCause(NamedTuple):
    cause: str
    why: str
    trip: bool  # an HSC tripped or a VR's power-good is low: the tray lost its 12 V, not just sagged
    new: dict[int, list[str]]  # HSC address -> flags set during the run


def tray_causes(power: TrayPower, before: TrayPower) -> dict[int, TrayCause]:
    """tray -> TrayCause for each tray whose reads explain a lost chip: POWER for an
    HSC that tripped or latched a fault during the run (a latch already set before it
    explains nothing); POWER-DOWNSTREAM for 12 V good on both HSCs with a VR's power-good low."""
    out = {}
    for t in HSC_BUS:
        hscs = {a: power.hsc.get((t, a)) for a in HSC_ADDRS}
        new = {}
        for a, h in hscs.items():
            pre = before.hsc.get((t, a))
            if h and (fresh := [f for f in h.flags if f not in (pre.flags if pre else ())]):
                new[a] = fresh
        tripped = any(_tripped(h) for h in hscs.values())
        bad = [f"HSC {a:#04x} {h}" for a, h in hscs.items() if _tripped(h) or new.get(a)]
        pg = power.vr_pg.get(t)
        if bad:
            out[t] = TrayCause("POWER", f"UBB{t} 12 V: {'; '.join(bad)}", tripped, new)
        elif all(hscs.values()) and pg is not None and pg != VR_PG_ALL:
            vins = ", ".join(f"{h.vin:.2f} V" for h in hscs.values() if h.vin is not None)
            out[t] = TrayCause("POWER-DOWNSTREAM", f"UBB{t} 12 V good ({vins}) but VR power-good {pg:#04x}", True, new)
    return out


def check_tray_power(phase: Phase, bmc: Bmc, before: TrayPower | None, keep: str) -> dict[int, TrayCause]:
    """Read the trays after the load, judge them against *before*, then clear the latches.

    One ``tray_power`` check per tray with a finding: FAIL for a tripped HSC, a low
    VIN or a VR power-good low, WARN for an under-voltage latch alone (the chips rode
    through the sag) or an HSC that did not answer; else one PASS. Then
    ``hsc_latch_clear``, which clears the latches unless *keep* says why not. Returns tray_causes() of the reads after the load.
    """
    after = read_power(bmc)
    if after is None or before is None:
        log(f"tray power: skipped: {bmc.why}")
        phase.add(Check(name=TRAY_POWER, status=SKIP, details=f"skipped: {bmc.why}", ip="board"))
        return {}
    for t in HSC_BUS:
        hscs = ", ".join(f"HSC {a:#04x} {after.hsc[(t, a)] or 'no answer'}" for a in HSC_ADDRS)
        pg = after.vr_pg[t]
        log(f"tray power UBB{t}: {hscs}, VR power-good {'not read' if pg is None else f'{pg:#04x}'}")
    if after.bricks:
        log("bricks: " + ", ".join(f"{n} {_brick(v, i)}" for n, (v, i) in after.bricks.items()))

    causes = tray_causes(after, before)
    found = False
    for t in HSC_BUS:
        notes, unread = [], []
        cause, why, trip, new = causes.get(t, TrayCause("", "", False, {}))
        for a in HSC_ADDRS:
            pre, post = before.hsc.get((t, a)), after.hsc[(t, a)]
            if post is None:
                unread.append(f"HSC {a:#04x} did not answer")
                continue
            held = [f for f in post.flags if f not in new.get(a, ())]
            if a in new:
                was = f"{pre.word:#06x}/{pre.inp:#04x}" if pre else "not read"
                notes.append(
                    f"HSC {a:#04x}: {', '.join(new[a])} set during the run "
                    f"(STATUS_WORD/STATUS_INPUT {was} -> {post.word:#06x}/{post.inp:#04x})"
                )
            if held:
                notes.append(f"HSC {a:#04x}: {', '.join(held)} already set before the run")
        if not (notes or cause or unread):
            continue
        found = True
        phase.add(
            Check(
                name=TRAY_POWER,
                status=FAIL if trip else WARN,
                details=f"UBB{t}: {'; '.join(notes + unread + ([why] if cause == 'POWER-DOWNSTREAM' else []))}",
                data={"before": _as_data(before, t), "after": _as_data(after, t)},
                ip="board",
                parts=_tray_parts(t),
                cause=cause,
            )
        )
    if not found:
        phase.add(
            Check(
                name=TRAY_POWER,
                status=PASS,
                details=f"{len(after.hsc)} tray HSCs: no trip, under-voltage latch or low VIN; every VR power-good high",
                data={"after": {t: _as_data(after, t) for t in HSC_BUS}, "bricks": after.bricks},
                ip="board",
            )
        )
    phase.add(_clear_latches(bmc, after, keep))
    return causes


def _brick(vout: float | None, iout: float | None) -> str:
    if vout is None and iout is None:
        return "no answer"
    return " ".join(x for x in ("" if vout is None else f"{vout:.2f} V", "" if iout is None else f"{iout:.1f} A") if x)


def _as_data(tp: TrayPower, tray: int) -> dict:
    return {
        "hsc": {f"{a:#04x}": (vars(h) if (h := tp.hsc.get((tray, a))) else None) for a in HSC_ADDRS},
        "vr_pg": tp.vr_pg.get(tray),
    }


def _clear_latches(bmc: Bmc, after: TrayPower, keep: str) -> Check:
    """CLEAR_FAULTS once to each HSC that answered, then STATUS_INPUT/STATUS_WORD re-read.
    Not sent at all when any HSC reads OFF: clearing it could turn that tray back on."""

    def done(status: str, details: str, **data) -> Check:
        log(f"HSC latch clear: {details}")
        return Check(name="hsc_latch_clear", status=status, details=details, data=data, ip="board")

    if keep:
        return done(SKIP, f"skipped: {keep}")
    off = [f"UBB{t} {a:#04x}" for (t, a), h in after.hsc.items() if h and "OFF" in h.flags]
    if off:
        return done(SKIP, f"skipped: HSC OFF ({', '.join(off)}); no CLEAR_FAULTS sent to any HSC", off=off)
    answered = [k for k, h in after.hsc.items() if h]
    sent = [k for k in answered if bmc.clear_faults(*k)]
    failed = [f"UBB{t} {a:#04x}" for t, a in answered if (t, a) not in sent]
    left = {f"UBB{t} {a:#04x}": h.flags for (t, a) in sent if (h := read_hsc(bmc, t, a)) and h.flags}
    still = "; ".join(f"{k} {', '.join(v)}" for k, v in left.items())
    return done(
        WARN if left or failed else PASS,
        f"CLEAR_FAULTS sent once to {len(sent)} HSC(s)"
        + (f", not taken by {', '.join(failed)}" if failed else "")
        + f"; after it: {f'still set: {still}' if left else 'all clear'}",
        sent=[f"UBB{t} {a:#04x}" for t, a in sent],
        failed=failed,
        still_set=left,
    )
