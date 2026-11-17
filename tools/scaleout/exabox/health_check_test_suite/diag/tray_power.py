# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 1b: tray 12 V feed (hot-swap controller and tray CPLD latches).

Reads only; see the comment block below for what is read and why."""

from __future__ import annotations

import os
import re
import subprocess
import time

from .common import emit_result, log, run
from .galaxy import TRAY_TO_UBB_NIBBLE
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

# A failing bus bar or tray 12 V feed sags the tray's input under load. Each
# tray has two MP5990 hot-swap controllers (HSCs) on its 12 V input, and they
# latch what they saw: STATUS_INPUT's VIN_UV_WARN / VIN_UV_FAULT, and
# STATUS_WORD's OFF and IOUT_OC when the sag or a trip turned the tray off. The
# tray CPLD latches which hot-swap's power-good dropped. A whole tray dropping
# off PCIe otherwise reads as 8 missing chips, with nothing pointing at the feed.
#
# The latches hold until the tray is re-powered (a power cycle, or the tray
# re-power of a glx_reset), so they are read here, right after the snapshot and
# before any reset or load. pre_reboot only: it runs no reset at all, and it is
# the tier run on a unit that has just misbehaved.
#
# Reads only, in-band through the BMC's I2C master (`ipmitool raw 0x06 0x52`).
# Every transaction goes through TRAY_POWER_ALLOWED; anything else raises before
# ipmitool is called. Nothing is ever written to an HSC: no CLEAR_FAULTS, no
# limit, no CLEAR_PEAK. The one write on the bus is the CPLD's debug-index
# select (0x50 <- 0x5F / 0x58) ahead of its 0x51 read, which only picks the
# register to read; it has no power or reset action.
#
# Each HSC is also read, for the record: the MP5990's FAULT_RECORD
# (FAh) and FAULT_RECORD_LAST (FBh), READ_VOUT (8Bh), IOUT_OC_FAULT_LIMIT (46h),
# IOUT_OC_WARN_LIMIT (4Ah), and READ_IOUT_PEAK (A6h) / READ_TEMP_PEAK (AFh).
# The two peaks clear when read, so each is read once, never retried; what they
# hold is the peak since the last peak read or power-on. Flags built on these
# are WARN only: a set FAULT_RECORD, or a feed's IOUT peak above its own
# IOUT_OC_WARN_LIMIT or more than HSC_PEAK_PCT % above the median of the other
# feeds' peaks.
#
# The phase does not gate yet: VIN_UV_WARN also latches on feeds that sag
# without dropping the tray, so it is worth recording before it is worth
# ticketing on.
# Tray -> its I2C bus behind the BMC. The BDF range follows UBB_NIBBLE_TO_TRAY.
TRAY_POWER_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}
TRAY_POWER_HSC_ADDRS = (0x40, 0x42)  # 8-bit addresses, as ipmitool takes them
TRAY_POWER_CPLD_ADDR = 0x80

HSC_STATUS_WORD, HSC_STATUS_IOUT, HSC_STATUS_INPUT = 0x79, 0x7B, 0x7C
HSC_FAULT_RECORD, HSC_FAULT_RECORD_LAST, HSC_READ_VOUT = 0xFA, 0xFB, 0x8B
HSC_IOUT_OC_FAULT_LIMIT, HSC_IOUT_OC_WARN_LIMIT = 0x46, 0x4A
HSC_READ_IOUT_PEAK, HSC_READ_TEMP_PEAK = 0xA6, 0xAF  # clear when read: read once
# (key, code) of the 2-byte words read per HSC after its status bytes.
HSC_RECORD_READS = (
    ("fault_record", HSC_FAULT_RECORD),
    ("fault_record_last", HSC_FAULT_RECORD_LAST),
    ("read_vout", HSC_READ_VOUT),
    ("iout_oc_fault_limit", HSC_IOUT_OC_FAULT_LIMIT),
    ("iout_oc_warn_limit", HSC_IOUT_OC_WARN_LIMIT),
)
HSC_PEAK_READS = (("iout_peak", HSC_READ_IOUT_PEAK), ("temp_peak", HSC_READ_TEMP_PEAK))
# MP5990 formats, as the Linux mp5990 driver and tray-test read them: VOUT
# 31.25 mV/LSB, IOUT (READ_IOUT_PEAK and the OC limits too) 62.5 mA/LSB,
# TEMP_PEAK in deg C.
HSC_VOUT_V_PER_LSB, HSC_IOUT_A_PER_LSB = 0.03125, 0.0625
HSC_PEAK_PCT = 20
CPLD_REG_ALIVE, CPLD_DBG_INDEX, CPLD_DBG_DATA = 0x1E, 0x50, 0x51
CPLD_DBG_PG, CPLD_DBG_POWER = 0x5F, 0x58

# (addr, bytes to read, bytes written) for every transaction the phase may send.
TRAY_POWER_ALLOWED = frozenset(
    [(a, 2, (HSC_STATUS_WORD,)) for a in TRAY_POWER_HSC_ADDRS]
    + [(a, 1, (HSC_STATUS_IOUT,)) for a in TRAY_POWER_HSC_ADDRS]
    + [(a, 1, (HSC_STATUS_INPUT,)) for a in TRAY_POWER_HSC_ADDRS]
    + [(a, 2, (code,)) for a in TRAY_POWER_HSC_ADDRS for _, code in HSC_RECORD_READS + HSC_PEAK_READS]
    + [
        (TRAY_POWER_CPLD_ADDR, 1, (CPLD_REG_ALIVE,)),
        (TRAY_POWER_CPLD_ADDR, 0, (CPLD_DBG_INDEX, CPLD_DBG_PG)),
        (TRAY_POWER_CPLD_ADDR, 0, (CPLD_DBG_INDEX, CPLD_DBG_POWER)),
        (TRAY_POWER_CPLD_ADDR, 1, (CPLD_DBG_DATA,)),
    ]
)

STATUS_WORD_BITS = {
    1: "CML",
    2: "TEMPERATURE",
    3: "VIN_UV",
    4: "IOUT_OC",
    5: "VOUT_OV",
    6: "OFF",
    7: "BUSY",
    8: "UNKNOWN",
    9: "OTHER",
    10: "FANS",
    11: "POWER_GOOD#",
    12: "MFR",
    13: "INPUT",
    14: "IOUT/POUT",
    15: "VOUT",
}
STATUS_INPUT_BITS = {
    0: "PIN_OP_WARN",
    1: "IIN_OC_WARN",
    2: "IIN_OC_FAULT",
    3: "UNIT_OFF_LOW_VIN",
    4: "VIN_UV_FAULT",
    5: "VIN_UV_WARN",
    6: "VIN_OV_WARN",
    7: "VIN_OV_FAULT",
}

# CPLD reg 0x1E / dbg 0x5F of a tray with both 12 V rails good, and of the
# known bus-bar signature (12 V-B power-good low, tray off).
CPLD_HEALTHY, CPLD_BUSBAR = ("ff", "7c"), ("00", "78")
CPLD_FEED_FAULTS = frozenset({"TRAY_12V_LOST", "BUSBAR_SIG", "HOTSWAP_OC"})
TRAY_POWER_HINT = "bus bar / tray 12 V feed"
TRAY_POWER_CALL_S = 15
# Whole-phase bound: an unresponsive BMC must not eat the run's own timeout.
TRAY_POWER_PHASE_S = 120
# Timeouts on one tray's bus before any read of it succeeded: skip the rest of
# that tray. Every tray's bus doing so (no read ok) means no route: stop.
TRAY_POWER_NO_ROUTE_TIMEOUTS = 2
TRAY_POWER_MAX_READS = 4
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


def _set_bits(value: int, names: dict) -> list[str]:
    return [name for bit, name in sorted(names.items()) if value >> bit & 1]


def decode_hsc_status(word: int, inp: int, iout: int | None = None) -> tuple[list[str], list[str]]:
    """(faults, set bits) for one HSC's STATUS_WORD / STATUS_INPUT (/ STATUS_IOUT).

    A fault is OFF, IOUT_OC, VIN_UV_FAULT or VIN_UV_WARN. The set-bit list
    names everything latched, for the report; CML alone is not a fault (a
    read of an unsupported code sets it).
    """
    set_bits = _set_bits(word, STATUS_WORD_BITS) + _set_bits(inp, STATUS_INPUT_BITS)
    if iout is not None and iout & 0x80:
        set_bits.append("IOUT_OC_FAULT")
    faults = []
    if word & 0x40:
        faults.append("OFF")
    if word & 0x10 or (iout is not None and iout & 0x80):
        faults.append("IOUT_OC")
    if inp & 0x10:
        faults.append("VIN_UV_FAULT")
    if inp & 0x20:
        faults.append("VIN_UV_WARN")
    return faults, set_bits


def decode_hsc_record(st: dict) -> dict:
    """Decoded record words of one hsc_record() read (None where not read)."""

    def scale(key: str, k: float) -> float | None:
        v = st.get(key)
        return None if v is None else round(v * k, 3)

    return {
        "vout_v": scale("read_vout", HSC_VOUT_V_PER_LSB),
        "iout_peak_a": scale("iout_peak", HSC_IOUT_A_PER_LSB),
        "temp_peak_c": st.get("temp_peak"),
        "iout_oc_fault_limit_a": scale("iout_oc_fault_limit", HSC_IOUT_A_PER_LSB),
        "iout_oc_warn_limit_a": scale("iout_oc_warn_limit", HSC_IOUT_A_PER_LSB),
    }


def hsc_record_text(st: dict) -> str:
    """One line of one HSC's record words, for the log.

    A read that failed or that the HSC does not support says 'not read', never 0.
    """
    d = decode_hsc_record(st)

    def h(key: str, w: int) -> str:
        v = st.get(key)
        return "not read" if v is None else f"0x{v:0{w}x}"

    def f(v, fmt: str, unit: str = "") -> str:
        return "not read" if v is None else format(v, fmt) + (f" {unit}" if unit else "")

    fl, wl = d["iout_oc_fault_limit_a"], d["iout_oc_warn_limit_a"]
    lims = f"{fl:.1f}/{wl:.1f} A" if fl is not None and wl is not None else f"{f(fl, '.1f', 'A')}/{f(wl, '.1f', 'A')}"

    return (
        f"FAULT_RECORD {h('fault_record', 4)}, FAULT_RECORD_LAST {h('fault_record_last', 4)}, "
        f"STATUS_IOUT {h('status_iout', 2)}, VOUT {f(d['vout_v'], '.2f', 'V')}, "
        f"IOUT peak {f(d['iout_peak_a'], '.1f', 'A')}, TEMP peak {f(d['temp_peak_c'], 'd', 'C')}, "
        f"IOUT OC fault/warn limit {lims}"
    )


def hsc_record_warnings(hscs_by_tray: dict) -> dict[int, list[str]]:
    """WARN lines per tray from the record words of every HSC ({tray: {addr: hsc_record()}}).

    A set FAULT_RECORD; an IOUT peak above the HSC's own IOUT_OC_WARN_LIMIT; or
    an IOUT peak more than HSC_PEAK_PCT % above the median of the other feeds'
    peaks (3 or more of them read). FAULT_RECORD_LAST has no rule: it reads
    0x0012 on every healthy HSC seen so far.
    """
    peaks = {
        (t, a): decode_hsc_record(st)["iout_peak_a"]
        for t, hscs in hscs_by_tray.items()
        for a, st in hscs.items()
        if st.get("iout_peak") is not None
    }
    out: dict[int, list[str]] = {t: [] for t in hscs_by_tray}
    for t, hscs in sorted(hscs_by_tray.items()):
        for a, st in sorted(hscs.items()):
            key = f"HSC 0x{a:02x}"
            if st.get("fault_record"):
                out[t].append(f"{key}: FAULT_RECORD 0x{st['fault_record']:04x} (FAh) set; WARN only")
            peak = peaks.get((t, a))
            if peak is None:
                continue
            lim = decode_hsc_record(st)["iout_oc_warn_limit_a"]
            if lim and peak > lim:
                out[t].append(
                    f"{key}: IOUT peak {peak:.1f} A (A6h, since the last peak read) above its own "
                    f"IOUT_OC_WARN_LIMIT {lim:.1f} A (4Ah); WARN only"
                )
            others = sorted(v for k, v in peaks.items() if k != (t, a))
            if len(others) >= 3:
                n = len(others)
                med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
                if med > 0 and peak > med * (1 + HSC_PEAK_PCT / 100):
                    out[t].append(
                        f"{key}: IOUT peak {peak:.1f} A, {(peak / med - 1) * 100:.0f}% above the median "
                        f"{med:.1f} A of the other {n} feeds (WARN over {HSC_PEAK_PCT}%); WARN only"
                    )
    return out


def decode_cpld_power(reg1e: str | None, dbg5f: str | None, dbg58: str | None = None) -> tuple[str, str]:
    """(state, text) for a tray CPLD's reg 0x1E / dbg 0x5F (/ dbg 0x58), as two-digit hex strings.

    dbg 0x5F bit 3 / bit 2 are the 12 V-A / 12 V-B hot-swap power-good; dbg
    0x58 bit 7 is power enable, bit 6 brake_n, bit 3 the over-current warning
    (active low). reg 0x1E reads ff while the tray is up.
    """
    if reg1e is None or dbg5f is None:
        return "UNREAD", "no stable read"

    def bit(v: str | None, b: int) -> bool | None:
        return None if v is None else bool(int(v, 16) >> b & 1)

    low = [rail for rail, b in (("A", 3), ("B", 2)) if not bit(dbg5f, b)]
    oc_n = bit(dbg58, 3)
    if (reg1e, dbg5f) == CPLD_HEALTHY:
        return "HEALTHY", "12 V rails good"
    if low and oc_n is False:
        return "HOTSWAP_OC", f"12 V-{'/'.join(low)} hot-swap power-good low with an over-current warning"
    if (reg1e, dbg5f) == CPLD_BUSBAR:
        return "BUSBAR_SIG", "12 V-B hot-swap power-good low, tray off: the bus-bar signature"
    if low and reg1e != "ff":
        return "TRAY_12V_LOST", f"12 V-{'/'.join(low)} hot-swap power-good low, tray off"
    if dbg5f == CPLD_HEALTHY[1] and reg1e != "ff":
        return "CHIPS_DOWN_12V_GOOD", "tray down with its 12 V good: not the 12 V feed"
    return "OTHER", "unexpected reading"


def _parse_raw_bytes(out: str, nbytes: int) -> list[int] | None:
    toks = out.split()
    if len(toks) != nbytes or not all(re.fullmatch(r"[0-9a-fA-F]{2}", t) for t in toks):
        return None
    return [int(t, 16) for t in toks]


class TrayPowerReader:
    """In-band reads of the tray HSCs and CPLDs, gated by TRAY_POWER_ALLOWED.

    `runner(argv) -> CompletedProcess` and `clock` are injectable so the tests
    can see every transaction without ipmitool. Once `budget_s` is spent, or
    the first reads only time out (no route), no more transactions are sent:
    `stopped` says why and every later read returns None.
    """

    def __init__(self, runner=None, clock=time.monotonic, budget_s=TRAY_POWER_PHASE_S):
        base = ["ipmitool"]
        if os.geteuid() != 0:
            base = ["sudo", "-n"] + base
        self.base = base
        self.clock = clock
        self.budget_s = budget_s
        self.deadline = clock() + budget_s
        self.runner = runner or self._run
        self.reads_ok = 0
        self.timeouts = 0
        self.bus_ok: dict[int, int] = {}
        self.bus_timeouts: dict[int, int] = {}
        self.errors: list[str] = []
        self.bus_err: dict[int, str] = {}  # last error per bus: the 'not read' reason of its tray
        self.stopped = ""

    def _err(self, bus: int, msg: str) -> None:
        self.errors.append(msg)
        self.bus_err[bus] = msg

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        left = self.deadline - self.clock()
        return run(argv, capture_output=True, text=True, timeout=max(1.0, min(TRAY_POWER_CALL_S, left)))

    def txn(self, bus: int, addr: int, nbytes: int, *payload: int) -> list[int] | None:
        if bus not in TRAY_POWER_BUS.values() or (addr, nbytes, tuple(payload)) not in TRAY_POWER_ALLOWED:
            raise ValueError(f"tray power: refusing bus 0x{bus:02x} addr 0x{addr:02x} {nbytes} {payload}")
        argv = self.base + ["raw", "0x06", "0x52", f"0x{bus:02x}", f"0x{addr:02x}", f"0x{nbytes:02x}"]
        argv += [f"0x{b:02x}" for b in payload]
        if not self.stopped and self.clock() >= self.deadline:
            self.stopped = f"phase time limit of {self.budget_s:g} s spent"
        if self.stopped:
            return None
        if self.bus_timeouts.get(bus, 0) >= TRAY_POWER_NO_ROUTE_TIMEOUTS and not self.bus_ok.get(bus):
            return None  # this tray's bus hangs: the next tray gets its own first reads
        try:
            cp = self.runner(argv)
        except subprocess.TimeoutExpired as e:
            self.timeouts += 1
            self.bus_timeouts[bus] = self.bus_timeouts.get(bus, 0) + 1
            self._err(bus, f"ipmitool: no reply in {e.timeout:g} s")
            hung = [b for b in TRAY_POWER_BUS.values() if self.bus_timeouts.get(b, 0) >= TRAY_POWER_NO_ROUTE_TIMEOUTS]
            if self.reads_ok == 0 and len(hung) == len(TRAY_POWER_BUS):
                self.stopped = "no route: ipmitool timed out"
            return None
        except FileNotFoundError as e:
            self._err(bus, repr(e))
            self.stopped = "no route: ipmitool not found"
            return None
        except (OSError, subprocess.SubprocessError) as e:
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
        if nbytes == 0:
            return []
        got = _parse_raw_bytes(cp.stdout or "", nbytes)
        if got is None:
            self._err(bus, f"unparsed reply {cp.stdout!r}")
        else:
            self.reads_ok += 1
            self.bus_ok[bus] = self.bus_ok.get(bus, 0) + 1
        return got

    def _read(self, bus: int, addr: int, nbytes: int, code: int) -> list[int] | None:
        return self.txn(bus, addr, nbytes, code) or self.txn(bus, addr, nbytes, code)

    @staticmethod
    def _word16(got: list[int] | None) -> int | None:
        return None if got is None else got[0] | got[1] << 8  # low byte first

    def hsc_status(self, bus: int, addr: int) -> dict:
        """One HSC's latched status (the FAIL-deciding reads), or the reads that failed. One retry per read."""
        word = self._read(bus, addr, 2, HSC_STATUS_WORD)
        inp, iout = self._read(bus, addr, 1, HSC_STATUS_INPUT), self._read(bus, addr, 1, HSC_STATUS_IOUT)
        return {
            "status_word": self._word16(word),
            "status_input": None if inp is None else inp[0],
            "status_iout": None if iout is None else iout[0],
        }

    def hsc_record(self, bus: int, addr: int, st: dict) -> dict:
        """Add one HSC's record and peak words (WARN only) to its hsc_status() `st`."""
        for key, code in HSC_RECORD_READS:
            st[key] = self._word16(self._read(bus, addr, 2, code))
        for key, code in HSC_PEAK_READS:
            st[key] = self._word16(self.txn(bus, addr, 2, code))  # clears on read: once, no retry
        return st

    def _stable(self, once) -> str | None:
        """Read until two in a row agree (the CPLD bus is shared with the BMC's own polling)."""
        prev = None
        for _ in range(TRAY_POWER_MAX_READS):
            v = once()
            if v is not None and v == prev:
                return v
            prev = v
        return None

    def cpld(self, bus: int) -> dict:
        def reg(r: int):
            got = self.txn(bus, TRAY_POWER_CPLD_ADDR, 1, r)
            return None if got is None else f"{got[0]:02x}"

        def dbg(r: int):
            if self.txn(bus, TRAY_POWER_CPLD_ADDR, 0, CPLD_DBG_INDEX, r) is None:
                return None
            return reg(CPLD_DBG_DATA)

        return {
            "reg1e": self._stable(lambda: reg(CPLD_REG_ALIVE)),
            "dbg5f": self._stable(lambda: dbg(CPLD_DBG_PG)),
            "dbg58": self._stable(lambda: dbg(CPLD_DBG_POWER)),
        }


def tray_power_check(tray: int, hscs: dict, cpld: dict, why: str = "", warns=()) -> Check:
    """One tray's Check from its HSC reads ({addr: hsc_record()}) and CPLD read (cpld()).

    `why` says why reads were cut short (the phase's time limit), for the
    'not read' text. A tray whose CPLD power enable reads low was powered off on
    purpose: its OFF latch is no 12 V feed fault, so it is a WARN with no hint.
    `warns` (hsc_record_warnings) are added to the details and turn a PASS into
    a WARN, never into a FAIL.
    """
    warns = list(warns)
    where = f"tray {tray} ({tray_bdf_range(tray)})"
    findings, unread, data = [], [], {"tray": tray, "bdf_range": tray_bdf_range(tray), "hsc": {}, "cpld": dict(cpld)}
    for addr, st in sorted(hscs.items()):
        key = f"0x{addr:02x}"
        word, inp, iout = st.get("status_word"), st.get("status_input"), st.get("status_iout")
        if word is None or inp is None:
            unread.append(f"HSC {key}")
            data["hsc"][key] = dict(st, **decode_hsc_record(st))
            continue
        faults, set_bits = decode_hsc_status(word, inp, iout)
        data["hsc"][key] = dict(st, faults=faults, bits=set_bits, **decode_hsc_record(st))
        if faults:
            findings.append(
                f"HSC {key}: {', '.join(faults)} latched (STATUS_WORD 0x{word:04x}, STATUS_INPUT 0x{inp:02x})"
            )
    state, text = decode_cpld_power(cpld.get("reg1e"), cpld.get("dbg5f"), cpld.get("dbg58"))
    data["cpld"].update(state=state, text=text)
    if state in CPLD_FEED_FAULTS:
        findings.append(f"CPLD: {state}: {text} (reg 0x1E={cpld['reg1e']} dbg 0x5F={cpld['dbg5f']})")
    elif state == "UNREAD":
        unread.append("CPLD")
    dbg58 = cpld.get("dbg58")
    powered_off = dbg58 is not None and not int(dbg58, 16) & 0x80
    data["cpld"]["power_enable"] = None if dbg58 is None else not powered_off

    name = f"tray{tray}_12v_feed"
    not_read = f"not read: {', '.join(unread)}" + (f" ({why})" if why else "")
    if powered_off:
        lines = [f"{where}: tray powered off (CPLD power enable low, dbg 0x58={dbg58}): latches not judged"]
        lines += findings + ([not_read] if unread else []) + warns
        return Check(name=name, status=WARN, details="\n".join(lines), data=data, ip="board")
    if findings:
        lines = [f"{where}: check {TRAY_POWER_HINT}"] + findings
        if unread:
            lines.append(not_read)
        return Check(name=name, status=FAIL, details="\n".join(lines + warns), data=data, ip="board")
    if unread:
        return Check(name=name, status=WARN, details="\n".join([f"{where}: {not_read}"] + warns), data=data, ip="board")
    if state == "OTHER":
        return Check(
            name=name, status=WARN, details="\n".join([f"{where}: CPLD {text}"] + warns), data=data, ip="board"
        )
    extra = f"; CPLD {text}" if state != "HEALTHY" else ""
    details = "\n".join([f"{where}: no 12 V latch set{extra}"] + warns)
    return Check(name=name, status=WARN if warns else PASS, details=details, data=data, ip="board")


def run_tray_power(phase: Phase, dry_run: bool, runner=None, clock=time.monotonic) -> None:
    """Read and judge every tray's 12 V feed latches into `phase`.

    Bounded by TRAY_POWER_PHASE_S: trays left when it is spent are WARN 'not
    read'. Every tray is tried, so a dead tray does not hide the others.
    No successful read on any tray means no route: one SKIP.
    """
    if dry_run:
        phase.add(Check(name="tray_12v_feed", status=SKIP, details="(dry) ipmitool raw 0x06 0x52 reads", ip="board"))
        return
    reader = TrayPowerReader(runner, clock)
    reads = {}
    # Pass 1: every tray's latches and CPLD, which decide FAIL. Pass 2: the record
    # and peak words, WARN only, so a slow BMC spends the phase limit on them last.
    for tray, bus in sorted(TRAY_POWER_BUS.items()):
        hscs = {addr: reader.hsc_status(bus, addr) for addr in TRAY_POWER_HSC_ADDRS}
        # No early exit on a tray with no reply: a dead tray NACKs fast and
        # must not hide the next ones; hangs are bounded by the reader.
        cpld = reader.cpld(bus)
        reads[tray] = (hscs, cpld, reader.stopped or reader.bus_err.get(bus, ""))
    for tray, bus in sorted(TRAY_POWER_BUS.items()):
        for addr, st in reads[tray][0].items():
            reader.hsc_record(bus, addr, st)
    # The peak rule compares each feed with all the others, so it runs once every tray is read.
    warns = hsc_record_warnings({t: r[0] for t, r in reads.items()})
    checks = [tray_power_check(t, hscs, cpld, why=why, warns=warns[t]) for t, (hscs, cpld, why) in reads.items()]
    for t, (hscs, _, _) in reads.items():
        for addr, st in sorted(hscs.items()):
            if st.get("status_word") is not None:
                log(f"tray_power: tray {t} HSC 0x{addr:02x}: {hsc_record_text(st)}")
    if reader.stopped:
        log(f"tray_power: stopped reading: {reader.stopped}")
    if reader.reads_ok == 0:
        # No route at all (no ipmitool, no sudo, no BMC): one SKIP, not four WARNs.
        first = reader.errors[0] if reader.errors else "no reply"
        phase.add(Check(name="tray_12v_feed", status=SKIP, details=f"not read: {first}", ip="board"))
        return
    for c in checks:
        phase.add(c)
        emit_result(c.name, c.status)
