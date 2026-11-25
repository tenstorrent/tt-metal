# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hardware captures around each stage and step: tray power, drops, hangs and PCIe errors.

``run(event, step, status, outdir, args)`` returns an event's FAIL and WARN lines; the rest goes to the log and to
``<outdir>/captures/`` (its state too, so separate ``python3 -m diag.captures`` runs work as one).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from . import aer, hsc
from .common import SUITE_DIR, log, resolve_optional_tool, root_cmd, run_with_timeout
from .galaxy import expected_bdf, parse_bdf_physical
from .hsc import HSC_ADDRS, HSC_BUS, INPUT_BITS, IOUT_LSB, VIN_LSB, WORD_BITS, Bmc, tray_bdfs

EVENTS = ("stage-start", "before", "after", "stage-end", "reset", "run")
LOAD_JUDGED_PCT = 80  # under this share of the planned load a clean HSC proves nothing
VIN_MIN_V = 11.0  # a loaded VIN under this: WARN
FEED_VIN_MIN_V = 11.15  # bus-bar stage: VIN floor of the marginal 12 V feed WARN
SAG_MOHM, SAG_INFO_MOHM = 13.0, 11.0  # 12 V sag (idle VIN - VIN) / IOUT: WARN over, said over
SAG_MIN_A, IDLE_MAX_A = 40, 30  # sag samples at or above; idle VIN from samples at or below
FEED_GAP_V = 0.25  # a feed's loaded VIN median this far under the other feeds' median: the weakest feed, WARN
PEAK_PCT = 20  # a feed's peak this far over the median of the others: WARN
OC_LIMIT_A = (50, 1000)  # a plausible IOUT_OC limit; outside: not read
POWER_S = 2  # power poll period
AER_S = 60  # corrected AER rate poll period
_MUX = ("raw", "0x06", "0x52", f"{hsc.BRICK_BUS:#04x}", f"{hsc.BRICK_MUX:#04x}")
BRICK_SUDO = ((*_MUX, "0x01"), (*_MUX, "0x00", "0x01"))  # the mux reads and writes the brick pass makes
BRICK_LOAD_S = 90  # the poller's brick pass under load (10-15 s of BMC reads): the HSC VIN sampler loses under a sixth
RESET_MASK_S = 40  # VR power-good falls this near a reset are the reset's
COLLAPSE_W, HOLD_W = 50, 600  # one tray at or under COLLAPSE_W while every other holds HOLD_W: its 12 V collapsed
CPLD_DUMP_MAX = 4
VCORE_HIGH_MV = 30  # a chip's mean core voltage this far over the box median of chip means: said
PCI = Path("/sys/bus/pci/devices")
ALL_BDFS = [expected_bdf(t, a) for t in HSC_BUS for a in range(1, 9)]


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("hardware captures")
    g.add_argument(
        "--hsc-clear", action="store_true", help="CLEAR_FAULTS once per HSC per run (none when any reads OFF)"
    )
    g.add_argument("--brick-mux", action="store_true", help="read the power bricks: mux channel select, read, restore")
    g.add_argument("--i2c-lock", help="lock file for in-band BMC I2C (default /tmp/hc-bmc-<uid>.lock)")
    g.add_argument("--site-data", help="site rows: 'brick_mux <host|*> <why>'")
    g.add_argument("--load-pct", type=float, default=100.0, help="share of the planned load the stage reached")
    g.add_argument("--hang-guard", action="store_true", help="take a hung chip off PCIe (root only)")
    g.add_argument(
        "--vcore-s", type=float, default=0.1, help="core voltage sample period in the heavy step (0.1 s least)"
    )
    g.add_argument("--cpld-dump", help="the vendor CPLD dump script (default: cpld-dump.sh on PATH)")
    g.add_argument("--ipmitool", help="the ipmitool command as is (default ipmitool, or sudo -n ipmitool)")


def _where(k: str) -> str:  # k: "<tray>:<addr>"
    return f"tray {k[0]} ({tray_bdfs(int(k[0]))}) HSC {k[2:]}"


def _hx(x) -> str:
    return "not read" if x is None else f"{x:#04x}"


def _f3(x, unit: str = "") -> str:
    return "-" if x is None else f"{x:.3f}{unit}"


def _hhmm(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _regs(bmc: Bmc, names, trays=HSC_BUS) -> dict[str, dict]:
    return {f"{t}:{a:#04x}": v for (t, a), v in hsc.read_regs(bmc, names, trays).items()} if bmc.cmd else {}


def _bits(r: dict) -> set[str]:
    return hsc._flags((r.get("STATUS_WORD") or 0, r.get("STATUS_INPUT") or 0))


def chips() -> list[str]:
    return [d.name for d in aer.tt_devices(PCI)]


def hung(bdf: str) -> bool:  # config space reads all ones: an MMIO read of the chip would stall the CPU
    try:
        return (PCI / bdf / "config").read_bytes()[:4] == b"\xff" * 4
    except OSError:
        return False


def _tray(bdf: str) -> int | None:
    return (parse_bdf_physical(bdf) or (None,))[0]


def _lines(path) -> list[str]:
    try:
        return Path(path).read_text().splitlines() if path else []
    except OSError:
        return []


def _brick_host(args) -> bool:
    host = socket.gethostname().split(".")[0]
    return any(
        r[:1] == ["brick_mux"] and len(r) > 1 and r[1] in (host, "*") for r in map(str.split, _lines(args.site_data))
    )


def _alive(pid: int, group: bool = False) -> bool:  # a process group: alive also when not ours to signal
    with suppress(ChildProcessError):
        if not group and os.waitpid(pid, os.WNOHANG)[0]:
            return False
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except PermissionError:
        return group
    except OSError:
        return False
    return True


def _save_cmd(path: Path, argv: list[str], what: str, timeout_s: float = 60) -> str:
    try:
        rc, out, err, _ = run_with_timeout(argv, timeout_s)
    except OSError as e:
        return f"{what} not saved: {e}"
    path.write_text(out + err)
    return f"{what} saved: {path}" + (f" (rc {rc})" if rc else "")


def _plaus(x) -> float | None:
    return float(x) if x is not None and OC_LIMIT_A[0] <= x <= OC_LIMIT_A[1] else None


def _run_start(bmc: Bmc, st: dict, args) -> list[str]:
    st["run"] = {"t": time.time()}
    names = ("STATUS_WORD", "STATUS_INPUT", "FAULT_RECORD", "FAULT_RECORD_LAST", "STATUS_IOUT", "READ_VOUT")
    if bmc.cmd:
        limits = ("IOUT_OC_FAULT_LIMIT", "IOUT_OC_WARN_LIMIT", *hsc.VIN_LIMITS, "READ_IOUT_PEAK")
        regs = st["run"]["regs"] = _regs(bmc, (*names, *limits))
    if not bmc.cmd or all(r["STATUS_WORD"] is None for r in regs.values()):
        log(
            f"HSC check not run: {bmc.why or 'no HSC answered'}; tray 12 V under-voltage latches and min VIN not judged"
        )
        return []
    pre = []
    for k, r in regs.items():
        sw, si = r["STATUS_WORD"], r["STATUS_INPUT"]
        log(f"HSC run start {_where(k)}: " + ", ".join(f"{n} {_hx(r[n])}" for n in names))
        log(f"HSC VIN limits {_where(k)}: " + ", ".join(f"{n} {hsc.vin_limit(n, r[n])}" for n in hsc.VIN_LIMITS))
        if b := _bits(r):
            pre.append(f"{_where(k)}: {'+'.join(sorted(b))} (STATUS_WORD {_hx(sw)}, STATUS_INPUT {_hx(si)})")
        if (sw or 0) & hsc.CML:
            pre.append(f"{_where(k)}: comm fault (CML), not power")
        p, w = r["READ_IOUT_PEAK"], _plaus(r["IOUT_OC_WARN_LIMIT"])
        if p is not None and w and p * IOUT_LSB >= w:
            pre.append(f"{_where(k)}: IOUT peak {p * IOUT_LSB:.1f} A at or above its own IOUT_OC_WARN_LIMIT {w:.0f} A")
    out = [f"HSC WARN host: {'; '.join(pre)}; latched before the run: pre-existing, not from this run"] if pre else []
    if not args.hsc_clear:
        return out
    if off := [_where(k) for k, r in regs.items() if (r["STATUS_WORD"] or 0) & 0x40]:
        log(f"HSC latch clear skipped: HSC OFF ({', '.join(off)}); no CLEAR_FAULTS sent to any HSC this run")
        return out
    sent = [k for k, r in regs.items() if r["STATUS_WORD"] is not None and bmc.clear_faults(int(k[0]), int(k[2:], 16))]
    st["run"]["cleared"] = after = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT"))
    left = [f"{k} {'+'.join(sorted(b))}" for k, r in after.items() if (b := _bits(r))]
    log(f"HSC latch: CLEAR_FAULTS sent once to {len(sent)} HSC(s); after it: {', '.join(left) or 'all clear'}")
    return out


def _vin_rows(d: Path, t0: float) -> dict[str, list[tuple[float, float | None]]]:
    """The poller's samples since t0 per HSC: (VIN V, IOUT A or None), in range only (6-15 V, IOUT up to 250 A)."""
    out: dict[str, list] = {}
    for f in (ln.split("\t") for ln in _lines(d / "vin.tsv")):
        if len(f) != 5 or float(f[0]) < t0 or not f[3].isdigit():
            continue
        v, i = int(f[3]) * VIN_LSB, int(f[4]) * IOUT_LSB if f[4].isdigit() else None
        if 6 <= v <= 15 and (i is None or i <= 250):
            out.setdefault(f"{f[1]}:{f[2]}", []).append((v, i))
    return out


def _judge(step: str, before: dict, after: dict, peaks: dict, samples: dict, st: dict, args, pset: dict) -> list[str]:
    out, info, named, sag, fails, loaded, refs = [], [], set(st.setdefault("named", [])), {}, {}, {}, {}
    # the bus-bar stage judges its feeds when its load was reached
    bb, low = step == "busbar", args.load_pct < LOAD_JUDGED_PCT
    feed = bb and not low
    lims = st["run"].get("regs", {})
    for k, a in after.items():
        b = before.get(k) or {}
        sw0, si0, sw1, si1 = b.get("STATUS_WORD"), b.get("STATUS_INPUT"), a["STATUS_WORD"], a["STATUS_INPUT"]
        s = samples.get(k, [])
        vins = [x * VIN_LSB for x in (b.get("READ_VIN"), a["READ_VIN"]) if x and 6 <= x * VIN_LSB <= 15]
        vins = [v for v, _ in s] + vins
        uv = [n for bit, n in INPUT_BITS.items() if None not in (si0, si1) and si1 & bit and not si0 & bit]
        wd = [n for bit, n in WORD_BITS.items() if None not in (sw0, sw1) and sw1 & bit and not sw0 & bit]
        uv, wd = [n for n in uv if f"{k}:{n}" not in named], [n for n in wd if f"{k}:{n}" not in named]
        if (si0 or 0) & 0x30:
            info.append(
                f"HSC {_where(k)} latched before load: a new dip on this tray cannot be seen until a power cycle clears it"
            )
        regs = ""
        if None not in (sw0, si0, sw1, si1):
            regs = f"STATUS_INPUT {si0:#04x} -> {si1:#04x}, STATUS_WORD {sw0:#06x} -> {sw1:#06x}"
        why = [f"{'+'.join(uv)} latched during {step} (clear before; {regs})"] if uv else []
        if "VIN_UV_WARN" in uv:
            lim = hsc.vin_limit("VIN_UV_WARN_LIMIT", lims.get(k, {}).get("VIN_UV_WARN_LIMIT"))
            pmin = f"poll min {min(v for v, _ in s):.2f} V over {len(s)} sample(s)" if s else "no poll sample"
            why.append(f"VIN fell below the UV warn limit {lim} (58h) at least once, depth unknown ({pmin})")
        if wd:
            why.append(f"{'+'.join(wd)} set during {step} ({regs})")
        if why:
            parts = "; per-part reads: " + ", then ".join(pset[k]) if pset.get(k) else ""
            out.append(fails.setdefault(k, f"HSC FAIL {_where(k)}: {'; '.join(why)}{parts}"))
            named |= {f"{k}:{n}" for n in uv + wd}
        idle = [v for v, i in s if i is not None and 2 < i <= IDLE_MAX_A]
        ref = None if b.get("READ_VIN") is None else b["READ_VIN"] * VIN_LSB
        ref = refs[k] = statistics.median(idle) if len(idle) >= 3 else ref
        iv = "idle VIN not read" if ref is None else f"idle VIN {ref:.2f} V"
        if not why and vins and min(vins) < (FEED_VIN_MIN_V if feed else VIN_MIN_V):
            what = f"min VIN {min(vins):.2f} V during {step}, below {VIN_MIN_V} V"
            if feed:  # the bus-bar stage's floor is the marginal 12 V feed line
                what = f"marginal 12 V feed: min VIN {min(vins):.2f} V, below {FEED_VIN_MIN_V:.2f} V ({iv}; bus-bar load) during {step}"
            out.append(f"HSC WARN {_where(k)}: {what} ({len(s)} sample(s)); WARN only, not a finding")
        load = [v for v, i in s if i is not None and i >= SAG_MIN_A]
        if load and ref is not None and not ((sw0 or 0) | (sw1 or 0)) & 0x40:  # an HSC that read OFF gives no number
            loaded[k] = statistics.median(load)
        if rs := [((ref - v) / i * 1000, v, i) for v, i in s if ref is not None and i is not None and i >= SAG_MIN_A]:
            r, v, i = max(rs)
            sag[k] = r
            line = f"{_where(k)}: 12 V sag {r:.1f} mOhm during {step} (VIN {v:.2f} V at IOUT {i:.1f} A, idle VIN {ref:.2f} V)"
            if r > SAG_MOHM:
                out.append(f"HSC WARN {line}, over {SAG_MOHM} mOhm; WARN only, not a finding")
            elif r > SAG_INFO_MOHM:
                info.append(f"HSC sag information {line}, over {SAG_INFO_MOHM} mOhm: information only, not a warn")
    pk = {k: r["READ_IOUT_PEAK"] * IOUT_LSB for k, r in peaks.items() if r.get("READ_IOUT_PEAK") is not None}
    n_before = len(out)
    for k, p in pk.items():
        w, f = (_plaus(lims.get(k, {}).get(n)) for n in ("IOUT_OC_WARN_LIMIT", "IOUT_OC_FAULT_LIMIT"))
        if w and p >= w:
            trip = f"trip IOUT_OC_FAULT_LIMIT {f:.1f} A (46h), " if f else "IOUT_OC_FAULT_LIMIT (46h) not read"
            if f and p > f:
                trip += f"the peak is {p - f:.1f} A over it"
            elif f and p == f:
                trip += "the peak is at its trip limit"
            elif f:
                trip += f"margin to the trip {f - p:.1f} A"
            head = f"HSC WARN {_where(k)}: IOUT peak {p:.1f} A during {step}, at or above its own IOUT_OC_WARN_LIMIT {w:.0f} A (4Ah)"
            if feed:
                iv = "not read" if refs.get(k) is None else f"{refs[k]:.2f} V"
                out.append(f"{head}, marginal 12 V feed, {trip}, idle VIN {iv}; WARN only, not a finding")
            else:
                out.append(f"{head}, {trip}; WARN only, not a tray verdict")
        others = [x for j, x in pk.items() if j != k]
        if len(others) >= 3 and (m := statistics.median(others)) > 0 and p > m * (1 + PEAK_PCT / 100):
            out.append(
                f"HSC WARN {_where(k)}: IOUT peak {p:.1f} A during {step}, {(p / m - 1) * 100:.0f}% above the median {m:.1f} A of the other {len(others)} feeds; WARN only, not a finding"
            )
    if bb and low and len(out) > n_before:  # a peak WARN fired
        info.append(
            f"HSC feed margin not judged: bus-bar load {args.load_pct:.0f}%, under {LOAD_JUDGED_PCT}% of the planned load; the peak WARNs above are the plain peak WARNs"
        )
    gaps = {
        k: statistics.median([x for j, x in loaded.items() if j != k]) - v for k, v in loaded.items() if len(loaded) > 1
    }
    if gaps and gaps[(k := max(gaps, key=gaps.get))] >= FEED_GAP_V - 1e-9:
        out.append(
            f"HSC WARN {_where(k)}: weakest feed during {step}: VIN median {loaded[k]:.2f} V at IOUT >= {SAG_MIN_A} A, {gaps[k]:.2f} V under {loaded[k] + gaps[k]:.2f} V, the median of the other {len(loaded) - 1} feed(s) (WARN at >= {FEED_GAP_V:.2f} V); WARN only, not a finding"
        )
    if sag:
        info.append(
            f"HSC sag after {step}: R max mOhm at IOUT >= {SAG_MIN_A} A "
            + " ".join(f"t{k.replace(':', '/')} {r:.1f}" for k, r in sag.items())
            + f" (WARN over {SAG_MOHM:.1f}, information over {SAG_INFO_MOHM:.1f})"
        )
    st["named"] = sorted(named)
    uvl = [k for k, a in after.items() if (a["STATUS_INPUT"] or 0) & 0x30]
    judged = "UV latched: " + (", ".join(uvl) or "none")
    if not uvl and low:
        judged = f"UV not judged: load not reached ({args.load_pct:.0f}% of the planned load)"
    vmin = ", ".join(f"{k} {min(v for v, _ in s):.2f}" for k, s in samples.items() if s)
    sags = ", ".join(f"{k} {r:.1f}" for k, r in sag.items()) or "none"
    info.append(f"HSC after {step}: {judged}; poll min VIN V {vmin or 'none'}; sag R max mOhm {sags}")
    ranked = sorted(sag.values(), reverse=True)
    for t in HSC_BUS:
        per = [[i for _, i in samples.get(f"{t}:{a:#04x}", []) if i is not None] for a in HSC_ADDRS]
        if tot := sorted(sum(x) for x in zip(*per)):  # the tray's HSCs summed per poll round
            info.append(
                f"HSC load tray {t} ({tray_bdfs(t)}) during {step}: IOUT mean {statistics.mean(tot):.1f} A, p90 {tot[int(0.9 * (len(tot) - 1))]:.1f} A ({len(tot)} poll(s))"
            )
    power: dict[str, list[str]] = {}  # the tray power line: a tray whose HSC turned off
    for k, ln in ((k, ln) for k, ln in fails.items() if "OFF" in ln.split(": ", 1)[1]):
        bits = " + ".join(sorted({"OFF", *(n for n in (*INPUT_BITS.values(), "IOUT_OC_FAULT") if n in ln)}))
        n = ranked.index(sag[k]) + 1 if k in sag else 0
        rank = f", sag {sag[k]:.1f} mOhm {'top' if n == 1 else f'rank {n}'} of {len(ranked)}" if n else ""
        power.setdefault(k[0], []).append(f"HSC {k[2:]} {bits}{rank}")
    for t, items in power.items():
        st.setdefault("tray_power", {})[t] = "; ".join(items)
        info.append(f"HSC power tray {t}: {'; '.join(items)}")
    for ln in info:
        log(ln)
    return out


def _bricks(bmc: Bmc, d: Path, when: str, args, idle: bool = False) -> None:
    """One brick pass. *idle*: the run-start pass, on every box, site row or not, that also reads each brick's ID."""
    if not idle and not _brick_host(args):
        host = socket.gethostname().split(".")[0]
        return log(
            f"WARNING brick reads skipped: no brick_mux row for {host} in --site-data {args.site_data or '(none)'}"
        )
    why = bmc.why
    if bmc.cmd and not all(bmc.listed(*a) for a in BRICK_SUDO):
        why = f"sudo -n -l does not list ipmitool {' '.join(BRICK_SUDO[0][:4])} (the brick mux)"
    if why:
        return log(f"brick reads skipped: {why}")
    modes, ident, bmc.failed = _brick_modes(d), {} if idle else None, ""
    timed_out = []

    def stop() -> bool:  # the idle pass runs on every box: one timed-out read ends it (the mux is still restored)
        if idle and bmc.failed == "ipmitool rc 124":
            timed_out.append(1)
        return bool(timed_out)

    rows, notes, fails = hsc.brick_pass(bmc, modes, stop=stop, ident=ident)
    (d / "brick_mode.json").write_text(json.dumps(modes))  # the exponents the poller's load pass reads Vout by
    _brick_rows(d, when, rows)
    if idle:
        return _brick_ids(d, when, rows, ident, notes, fails)
    bricks = ", ".join(f"{r[0]} {_f3(r[1], ' V')} {_f3(r[2], ' A')}" for r in rows)
    log(f"bricks and the 54 V rail {when}: {bricks}" + "".join(f"; {n}" for n in notes + fails))


def _brick_ids(d: Path, when: str, rows, ident: dict, notes: list[str], fails: list[str]) -> None:
    """One line per brick (ID fields, Vout) and one for the 54 V rail (with the pass's notes), to the report and
    captures/brick_id.txt; one WARNING line for the reads that failed."""
    got = {r[0]: r for r in rows}  # a pass that stopped early has no row for the rest: those say "not read"
    lines = [
        f"brick {n} {when}: "
        + ", ".join(f"{k} {v if v is not None else 'not read'}" for k, v in ident.get(n, {"MFR_*": None}).items())
        + f", Vout {_f3(got[n][1] if n in got else None, ' V')}"
        for n in hsc.BRICKS
    ] + [f"54 V rail {when}: {_f3(got['FAN'][1] if 'FAN' in got else None, ' V')}" + "".join(f"; {n}" for n in notes)]
    (d / "brick_id.txt").write_text("".join(f"{ln}\n" for ln in lines))
    for ln in lines:
        log(ln)
    if fails:  # one WARN line for the pass; the run goes on
        log(f"WARNING brick reads {when}: {'; '.join(fails)}")


def _brick_modes(d: Path) -> dict:
    with suppress(OSError, ValueError):
        return json.loads((d / "brick_mode.json").read_text())
    return {}


def _brick_rows(d: Path, when: str, rows) -> None:
    t = f"{time.time():.0f}"  # one stamp per pass: _brick_load counts passes by it
    with open(d / "brick.tsv", "a") as fh:
        fh.writelines("\t".join([t, when, r[0], *map(_f3, r[1:])]) + "\n" for r in rows)


def _brick_load(d: Path, step: str, t0: float) -> str | None:
    """The poller's brick passes during *step* since *t0*: the lowest Vout per brick and of the 54 V rail."""
    lo: dict[str, float] = {}
    n = set()
    for r in (r.split("\t") for r in _lines(d / "brick.tsv")):
        if len(r) > 3 and r[1] == f"during {step}" and float(r[0]) >= t0:
            n.add(r[0])
            with suppress(ValueError):
                lo[r[2]] = min(lo.get(r[2], 99.0), float(r[3]))
    if n:
        v = ", ".join(f"{k} {x:.3f} V" for k, x in sorted(lo.items())) or "none read"
        return f"bricks during {step}: {len(n)} pass(es), every {BRICK_LOAD_S} s; min Vout {v}"
    return None


def _dump_tray(bmc: Bmc, d: Path, t: int, what: str) -> str:
    """The tray's HSC and CPLD power state (alive, thermal trip, VR power-good) before a reset clears it, to a file."""
    f = d / f"drop_{time.strftime('%Y%m%dT%H%M%S')}_tray{t}.txt"
    if not bmc.cmd:
        return f"tray {t} ({tray_bdfs(t)}) power state not read: {bmc.why}"
    regs = _regs(bmc, [n for n in hsc.REG if "PEAK" not in n and "OC_" not in n and n != "READ_PIN"], trays=(t,))
    cp = {n: bmc.cpld(t, r) for n, r in (("alive", 0x1E), ("therm_trip", 0x2D), ("vr_pg", 0x19))}
    head = f"tray {t} ({tray_bdfs(t)}) power state at {what}"
    lines = [f"{head}, read only", *(f"cpld {n} {_hx(v)}" for n, v in cp.items())]
    lines += [f"hsc {k[2:]} {n} {'not read' if v is None else hex(v)}" for k, r in regs.items() for n, v in r.items()]
    f.write_text("\n".join(lines) + "\n")
    hs = "; ".join(
        f"HSC {k[2:]} not read"
        if r["STATUS_WORD"] is None
        else f"HSC {k[2:]} STATUS_WORD {r['STATUS_WORD']:#06x} VIN {(r['READ_VIN'] or 0) * VIN_LSB:.2f} V IOUT {(r['READ_IOUT'] or 0) * IOUT_LSB:.1f} A"
        for k, r in regs.items()
    )
    cpld = " ".join(f"{n} {'-' if v is None else f'{v:#04x}'}" for n, v in cp.items())
    return f"{head}: CPLD {cpld}; {hs} (dump {f})"


def _cpld_dump(bmc: Bmc, st: dict, d: Path, trays: list[int], args) -> str:
    """The vendor CPLD dump, once per drop event, at most CPLD_DUMP_MAX a run; each tray CPLD's index 0x50 restored."""
    tool, why = resolve_optional_tool(args.cpld_dump, "cpld-dump.sh", "--cpld-dump", "not installed")
    if not tool:
        return f"CPLD dump: {why}"
    if trays == st.get("cpld_open") or st.get("cpld_n", 0) >= CPLD_DUMP_MAX:
        return f"CPLD dump of tray(s) {trays}: not run (same event, or {CPLD_DUMP_MAX} dumps this run)"
    st["cpld_open"], st["cpld_n"] = trays, st.get("cpld_n", 0) + 1
    runs = {m: root_cmd(["timeout", "-s", "KILL", "90", "bash", tool, m])[0] for m in ("DIAG", "PSU")}
    if not bmc.cmd:
        return f"CPLD dump not run: {bmc.why}"
    if None in runs.values():
        return f"CPLD dump not run: no root, and sudo -n -l does not list 'timeout -s KILL 90 bash {tool} DIAG|PSU'"
    f = d / "cpld" / f"{time.strftime('%Y%m%dT%H%M%S')}_tray{'+'.join(map(str, trays))}.txt"
    f.parent.mkdir(exist_ok=True)
    with bmc.locked():
        idx = {t: bmc.i2c(HSC_BUS[t], hsc.TRAY_CPLD, 0x50, 1, lock=False) for t in HSC_BUS}
        text = "".join(f"=== {m}\n" + "".join(run_with_timeout(argv, 100, cwd=d)[1:3]) for m, argv in runs.items())
        for t, v in idx.items():
            if v is not None:
                bmc.raw(0x06, 0x52, HSC_BUS[t], hsc.TRAY_CPLD, 0x00, 0x50, v, lock=False)
    f.write_text(
        text
        + "".join(
            f"tray {t} CPLD index 0x50 before: {'not read' if v is None else hex(v)}, written back\n"
            for t, v in idx.items()
        )
    )
    return f"CPLD dump saved: {f}"


def _evidence(bmc: Bmc, st: dict, d: Path, what: str, trays: list[int], args) -> None:
    """Before anything resets what it shows: each tray's power dump, the vendor CPLD dump, the SEL and the kernel log."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    lines = [_dump_tray(bmc, d, t, what) for t in trays] + ([_cpld_dump(bmc, st, d, trays, args)] if trays else [])
    if not bmc.cmd:
        lines.append(f"SEL not saved: {bmc.why}")
    elif bmc.listed("sel", "elist"):
        lines.append(_save_cmd(d / f"sel_{ts}.txt", [*bmc.cmd, "sel", "elist"], "SEL"))
    else:
        lines.append("SEL not saved: no root, and sudo -n -l does not list ipmitool sel elist")
    kern = root_cmd(["journalctl", "-k", "-b", "--no-pager"])[0]
    no_kern = "kernel log not saved: no root, and sudo -n -l does not list journalctl -k -b --no-pager"
    lines.append(_save_cmd(d / f"kern_{ts}.txt", kern, "kernel log") if kern else no_kern)
    for ln in lines:
        log(f"{what}: {ln}")


def _remove(bdf: str, args) -> str:
    if not args.hang_guard:
        return "not removed (--hang-guard off)"
    if os.geteuid() != 0:
        return "not removed: needs root"
    try:
        (PCI / bdf / "remove").write_text("1")
        return "removed from PCIe"
    except OSError as e:
        return f"not removed: {e}"


def _guard(st: dict, d: Path, step: str, args) -> None:
    """Name the hung chips the poller saw (and took off PCIe with --hang-guard) and any found now."""
    seen = {}
    with suppress(ValueError):
        for ln in _lines(d / "hung.tsv"):
            t, b, how = ln.split("\t", 2)
            seen.setdefault(b, (float(t), how))
    for b in (st.get("stage") or {}).get("chips", []):
        if b not in seen and hung(b):
            seen[b] = (time.time(), _remove(b, args))
    for b, (t, how) in seen.items():
        log(
            f"hang guard: {b} (tray {_tray(b)}) config space reads ffffffff, first seen {_hhmm(t)} ({time.time() - t:.0f} s before the end of {step}); {how}"
        )


def _drops(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> None:
    """Chips gone since the stage started (else of the 32): each tray's evidence once; a bad step's SEL and kernel log."""
    gone = sorted(set(st["stage"]["chips"] if "stage" in st else ALL_BDFS) - set(chips()))
    trays = sorted({_tray(b) for b in gone} - set(st.get("dropped", [])))
    if trays:
        st["dropped"] = sorted({*st.get("dropped", []), *trays})
        log(
            f"chips lost during {step}: {', '.join(b.split(':', 1)[1] for b in gone)} (tray {', '.join(map(str, trays))})"
        )
    if trays or status.lower() in ("hung", "hang", "fail", "drop", "lost"):
        _evidence(bmc, st, d, f"the end of {step}", trays, args)


def _hwmon(bdf: str) -> tuple[float | None, float | None]:
    out = []
    for f, scale in (("power1_input", 1e-6), ("curr1_input", 1e-3)):
        try:
            out.append(int(next((PCI / bdf / "hwmon").glob(f"hwmon*/{f}")).read_text()) * scale)
        except (OSError, StopIteration, ValueError):
            out.append(None)
    return tuple(out)


class _AerRate:
    """Each root port's corrected AER errors a minute, the larger of sysfs and the kernel log: aer_min.tsv rows."""

    def __init__(self):
        self.ports, self.kern = aer.root_ports(PCI), root_cmd(["dmesg"])[0] or ["dmesg"]
        self.t, self.cor, self.kt = time.time(), {p: aer.port_corrected(p, c) for p, c in self.ports.items()}, None
        self.kernel()

    def kernel(self) -> dict[str, list[int]]:
        try:
            out = run_with_timeout(self.kern, 20)[1]
        except OSError:
            return {}
        rows = [(float(m.group(1)), m.group(2)) for m in re.finditer(r"^\[\s*(\d+\.\d+)\]\s?(.*)$", out, re.M)]
        new = aer.kernel_events(ln for t, ln in rows if self.kt is not None and t > self.kt)
        self.kt = max([t for t, _ in rows] + ([self.kt] if self.kt is not None else []), default=None)
        return new

    def sample(self, f) -> None:
        now, kern = time.time(), self.kernel()
        for p, cs in self.ports.items():
            c, was = aer.port_corrected(p, cs), self.cor.get(p)
            k = sum(kern.get(x.name, [0])[0] for x in (p, *cs))
            rise = max(c - was if None not in (c, was) and c >= was else 0, k)
            per_min = rise * 60 / max(now - self.t, 60)
            f.write(f"{now:.0f}\t{p.name}\t{','.join(x.name for x in cs)}\t{rise}\t{per_min:.0f}\n")
            self.cor[p] = c
        self.t = now


def _aer_rate(d: Path, t0: float, step: str) -> list[str]:
    """A root port whose corrected AER errors rose over RATE_PER_MIN in any minute of the stage: a weak PCIe link."""
    seen: dict[str, list] = {}
    rows = [r.split("\t") for r in _lines(d / "aer_min.tsv")]
    for ts, port, cs, rise, pm in (r for r in rows if len(r) == 5 and float(r[0]) >= t0):
        s = seen.setdefault(port, [cs, 0, 0, 0, 0])  # chips, minutes, minutes over, peak, sum
        s[1:] = [s[1] + 1, s[2] + (int(pm) > aer.RATE_PER_MIN), max(s[3], int(pm)), s[4] + int(rise)]
    return [
        f"AER WARN weak PCIe link {', '.join(f'{b[5:]} (tray {_tray(b)})' for b in cs.split(','))}: root port {port} "
        f"corrected AER peak {peak}/min (over {aer.RATE_PER_MIN}/min in {over} of {n} min, +{tot} in all) during {step}"
        for port, (cs, n, over, peak, tot) in seen.items()
        if over
    ]


def poll(d: Path, watch: int, args, step: str) -> None:
    """Until stage-end's stop file, or the caller's process group *watch* is gone: each HSC's VIN (IOUT every other
    sweep) to vin.tsv; every POWER_S the BMC power sensors and each tray's VR power-good byte to power.tsv, each chip's
    power and current to chips.tsv, a newly hung chip to hung.tsv; every AER_S the AER rate to aer_min.tsv."""
    bmc, stop, bdfs, sweep, nxt = Bmc(args.ipmitool, args.i2c_lock), d / "poll.stop", chips(), 0, 0.0
    rate, nxt_aer = _AerRate(), time.time() + AER_S
    bricks = bool(args.brick_mux and bmc.cmd and _brick_host(args))
    nxt_brick, modes, said = time.time() + BRICK_LOAD_S, _brick_modes(d), set()
    signal.signal(
        signal.SIGTERM, lambda *_: sys.exit(143)
    )  # a stop past its wait: the brick pass still restores its mux
    files = [open(d / f"{n}.tsv", "a") for n in ("vin", "power", "chips", "hung", "aer_min")]
    fv, fp, fc, fh, fa = files
    while not stop.exists() and _alive(watch, group=True):
        for t in HSC_BUS if bmc.cmd else ():
            for a in HSC_ADDRS:
                v = bmc.hsc(t, a, "READ_VIN")
                i = bmc.hsc(t, a, "READ_IOUT") if v is not None and sweep % 2 == 0 else None
                if v is not None:
                    fv.write(f"{time.time():.0f}\t{t}\t{a:#04x}\t{v}\t{'-' if i is None else i}\n")
        sweep += 1
        if time.time() >= nxt:
            nxt = time.time() + POWER_S
            if bmc.cmd:
                pg = ["ERR" if v is None else f"{v:02x}" for v in (bmc.cpld(t, 0x19) for t in HSC_BUS)]
                s = [bmc.sensor(n) for n in hsc.SENSORS]
                fp.write(" ".join([f"{time.time():.3f}", *s[:5], *pg, *s[5:]]) + "\n")
            for b in list(bdfs):
                if hung(b):
                    bdfs.remove(b)
                    fh.write(f"{time.time():.0f}\t{b}\t{_remove(b, args)}\n")
                elif None not in (w := _hwmon(b)):
                    fc.write(f"{time.time():.0f}\t{b}\t{w[0]:.1f}\t{w[1]:.1f}\n")
            if time.time() >= nxt_aer:
                nxt_aer = time.time() + AER_S
                rate.sample(fa)
            for f in files:
                f.flush()
        if bricks and time.time() >= nxt_brick:
            rows, notes, fails = hsc.brick_pass(bmc, modes, load=True, stop=stop.exists)
            nxt_brick = time.time() + BRICK_LOAD_S
            _brick_rows(d, f"during {step}", rows)
            for n in notes + fails:
                if n not in said:  # a fault that stays is said once per stage, not on every pass
                    said.add(n)
                    log(n)
        time.sleep(0.05)
    rate.sample(fa)  # the stage's last minute
    for f in files:
        f.close()


def vcore(d: Path, watch: int, args) -> None:
    """Until the heavy step's end: each chip's core voltage (hwmon in0_input, mV) every --vcore-s (0.1 s at least) and
    its AICLK (tt_aiclk) every 10th sample to vcore.tsv. Read-only sysfs, no device open; a hung chip is not read again.
    """
    paths = {
        b: [next(g, None) for g in ((PCI / b).glob("hwmon/hwmon*/in0_input"), (PCI / b).glob("tenstorrent/*/tt_aiclk"))]
        for b in chips()
    }
    stop, n = d / "vcore.stop", 0
    with open(d / "vcore.tsv", "w") as f:
        while not stop.exists() and _alive(watch, group=True):
            for b, (mv, clk) in list(paths.items()):
                if n % 10 == 0 and hung(b):
                    del paths[b]
                    continue
                with suppress(OSError, ValueError, AttributeError):
                    aiclk = clk.read_text().strip() if clk and n % 10 == 0 else ""
                    f.write(f"{time.time():.1f}\t{b}\t{int(mv.read_text())}\t{aiclk}\n")
            n += 1
            f.flush()
            time.sleep(max(args.vcore_s, 0.1))


def vcore_lines(d: Path) -> list[str]:
    """The heavy step's core voltage: chips with a NUMERIC FAIL or WARN one line each, the others one summary line; one
    line per tray (mean [min-max] mV @ median AICLK MHz), one box line, and VCORE HIGH for a chip VCORE_HIGH_MV over the
    box median. Report only."""
    mv: dict[str, list[int]] = {}
    clk: dict[str, list[int]] = {}
    for r in (ln.split("\t") for ln in _lines(d / "vcore.tsv")):
        mv.setdefault(r[1], []).append(int(r[2]))
        if r[3].isdigit() and 0 < int(r[3]) < 5000:
            clk.setdefault(r[1], []).append(int(r[3]))
    if not mv:
        return ["NUMERIC VCORE: no samples (numeric verify: heavy chunk)"]
    bdf = r"\b([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\b"
    hits = {
        b
        for ln in _lines(d.parent / "findings.txt")
        if re.match(r"NUMERIC (FAIL|WARN) ", ln)
        for b in re.findall(bdf, ln)
    }
    out = [
        f"NUMERIC VCORE {b[5:]} (tray {_tray(b)}): min {min(v)} mV, mean {statistics.mean(v):.0f} mV over {len(v)} samples, numeric hit (numeric verify: heavy chunk)"
        for b, v in sorted(mv.items())
        if b[5:] in hits
    ]
    if clean := {b: v for b, v in mv.items() if b[5:] not in hits}:
        x = [y for v in clean.values() for y in v]
        out.append(
            f"NUMERIC VCORE {len(clean)} chips with no numeric hit: min {min(x)} mV, mean {statistics.mean(x):.0f} mV (numeric verify: heavy chunk)"
        )
    means = {b: round(statistics.mean(v)) for b, v in mv.items()}
    ck = {b: round(statistics.median(clk[b])) for b in mv if b in clk}
    for t in sorted({_tray(b) for b in mv}, key=str):
        cells = ", ".join(
            f"{b[5:]} {means[b]} [{min(mv[b])}-{max(mv[b])}] @{ck.get(b, '-')}" for b in sorted(mv) if _tray(b) == t
        )
        out.append(f"VCORE tray {t} under load (heavy chunk; mean [min-max] mV @ AICLK MHz): {cells}")
    med = statistics.median(means.values())
    mck = f"{statistics.median(ck.values()):.0f} MHz" if ck else "not read"
    out.append(
        f"VCORE box under load: {len(means)} chips, median of chip means {med:.0f} mV (range {min(means.values())}-{max(means.values())}), median AICLK {mck}"
    )
    out += [
        f"VCORE HIGH {b[5:]} (tray {_tray(b)}): mean {m} mV under load, {m - med:.0f} mV over the box median {med:.0f} mV (flag at +{VCORE_HIGH_MV} mV), AICLK {ck.get(b, 'not read')}; report only"
        for b, m in sorted(means.items(), key=lambda x: (-x[1], x[0]))
        if len(means) >= 3 and m - med >= VCORE_HIGH_MV
    ]
    return out


def _poll_start(st: dict, d: Path, args, mode: str = "poll", step: str = "") -> None:
    (d / f"{mode}.stop").unlink(missing_ok=True)
    argv = [sys.executable, "-m", "diag.captures", mode, str(d), *([step] if mode == "poll" else [])]
    argv += ["--watch-pgid", str(os.getpgrp())]
    argv += ["--vcore-s", str(args.vcore_s)] if mode == "vcore" else []
    argv += [x for k, v in (("--ipmitool", args.ipmitool), ("--i2c-lock", args.i2c_lock)) if v for x in (k, v)]
    argv += ["--hang-guard"] if args.hang_guard else []
    argv += ["--brick-mux", "--site-data", args.site_data] if args.brick_mux and args.site_data else []
    with open(d / f"{mode}.err", "a") as err:
        p = subprocess.Popen(
            argv, cwd=SUITE_DIR, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err
        )
    st[f"{mode}_pid"] = p.pid


def _poll_stop(st: dict, d: Path, mode: str = "poll") -> None:
    if not (pid := st.pop(f"{mode}_pid", None)):
        return
    (d / f"{mode}.stop").touch()
    for _ in range(150):  # one HSC sweep at most
        if not _alive(pid):
            return
        time.sleep(0.1)
    with suppress(OSError):
        os.kill(pid, 15)


def _power_rows(d: Path, t0: float, resets) -> list[list[str]]:
    """The power poll's rows since *t0*, away from resets."""
    rows = [r for r in map(str.split, _lines(d / "power.tsv")) if len(r) >= 10 and float(r[0]) >= t0]
    return [r for r in rows if all(abs(float(r[0]) - x) > RESET_MASK_S for x in resets)]


def vr_pg_fell(d: Path, t0: float, rows=None) -> dict[int, tuple[str, str]]:
    """Each tray whose VR power-good byte (its CPLD) read not ff on two polls in a row since *t0*, away from resets:
    the first such poll's (time, byte)."""
    if rows is None:
        resets = []
        with suppress(OSError, ValueError):
            resets = json.loads((d / "state.json").read_text()).get("resets", [])
        rows = _power_rows(d, t0, resets)
    out = {}
    for t in HSC_BUS:
        run = []
        for r in rows:
            v = r[5 + t].lower()
            run = [] if v == "ff" else run + [(r[0], v)] if len(v) == 2 else run
            if len(run) >= 2:
                out[t] = run[0]
                break
    return out


def _power_lines(d: Path, s: dict, step: str, st: dict) -> list[str]:
    """VR power-good falls (vr_pg_fell) and 12 V collapses (one tray at or under COLLAPSE_W while every other holds
    HOLD_W) during the stage, away from resets; each chip's peak power and current.
    """
    out, t0 = [], s["t0"]
    rows = _power_rows(d, t0, st.get("resets", []))
    pg = vr_pg_fell(d, t0, rows)
    for t in HSC_BUS:
        if fell := pg.get(t):
            out.append(
                f"12V FAIL tray {t} ({tray_bdfs(t)}): vr_pg {fell[1]} from {_hhmm(float(fell[0]))} during {step} (not ff on two polls in a row)"
            )
        for ts, p in ((r[0], [int(x) if x.isdigit() else None for x in r[1:5]]) for r in rows):
            if None not in p and p[t - 1] <= COLLAPSE_W and all(x >= HOLD_W for j, x in enumerate(p) if j != t - 1):
                out.append(
                    f"12V FAIL tray {t} ({tray_bdfs(t)}): Power_UBB{t - 1} {p[t - 1]} W at {_hhmm(float(ts))} while every other tray held {HOLD_W} W or more during {step}"
                )
                break
    peak: dict[str, list[float]] = {}
    with suppress(ValueError):
        for ln in _lines(d / "chips.tsv"):
            ts, b, w, a = ln.split("\t")
            if float(ts) >= t0:
                p = peak.setdefault(b, [0.0, 0.0])
                p[0], p[1] = max(p[0], float(w)), max(p[1], float(a))
    if peak:
        f = d / f"chip_peak_{step.replace(' ', '_').replace('/', '_')}.tsv"
        f.write_text("".join(f"{b}\t{_tray(b)}\t{w:.1f}\t{a:.1f}\n" for b, (w, a) in sorted(peak.items())))
        bw, ba = max(peak, key=lambda b: peak[b][0]), max(peak, key=lambda b: peak[b][1])
        log(
            f"peak per chip during {step}: {peak[bw][0]:.1f} W ({bw}, tray {_tray(bw)}), {peak[ba][1]:.1f} A ({ba}, tray {_tray(ba)}); per chip in {f}"
        )
    return out


def load_line(d: Path, t0: float, step: str) -> str:
    """The power poll since *t0* as one 'LOAD <step>: ...' line (also in load.txt): the four trays' sum (Power_UBB0-3)
    mean and peak, each tray's mean, the outlet air's peak and the hottest chip's peak power; "" with no poll row."""
    num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None  # noqa: E731
    rows = [[num(x) for x in r[1:16]] for r in _power_rows(d, t0, []) if len(r) >= 16]
    col = lambda i: [r[i] for r in rows if r[i] is not None]  # noqa: E731
    if not (trays_w := [sum(r[:4]) for r in rows if None not in r[:4]]):
        return ""
    trays = "/".join(f"{statistics.mean(c):.0f}" if (c := col(i)) else "-" for i in range(4))
    air = f"; outlet air peak {max(c):.0f} C" if (c := col(14)) else ""
    peak: dict[str, float] = {}
    with suppress(ValueError):
        for ln in _lines(d / "chips.tsv"):
            ts, b, w, _ = ln.split("\t")
            if float(ts) >= t0:
                peak[b] = max(peak.get(b, 0.0), float(w))
    top = max(peak, key=peak.get) if peak else None
    chip = f"; peak chip {peak[top]:.0f} W ({top}, tray {_tray(top)})" if top else ""
    line = (
        f"LOAD {step}: tray sum mean {statistics.mean(trays_w) / 1000:.2f} kW, peak {max(trays_w) / 1000:.2f} kW; "
        f"tray 1/2/3/4 mean {trays} W{air}{chip}"
    )
    with open(d / "load.txt", "a") as f:
        f.write(line + "\n")
    return line


# ── events ───────────────────────────────────────────────────────────────────


def _stage_start(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    _poll_stop(st, d)  # a stage whose stage-end never came
    before = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT", "READ_VIN", "READ_IOUT"))
    _regs(bmc, ("READ_IOUT_PEAK", "READ_TEMP_PEAK"))  # read-clear: the read after the stage is the stage's own peak
    num = lambda x, lsb, unit: "not read" if x is None else f"{x * lsb:.2f} {unit}"  # noqa: E731
    for k, r in before.items():
        log(
            f"HSC idle {_where(k)} before {step}: VIN {num(r['READ_VIN'], VIN_LSB, 'V')}, IOUT {num(r['READ_IOUT'], IOUT_LSB, 'A')}"
        )
    if vin := {k: r["READ_VIN"] * VIN_LSB for k, r in before.items() if r["READ_VIN"] is not None}:
        lo, hi = min(vin, key=vin.get), max(vin, key=vin.get)
        log(
            f"HSC idle box before {step}: VIN median {statistics.median(vin.values()):.3f} V, min {vin[lo]:.3f} V ({lo}), max {vin[hi]:.3f} V ({hi})"
        )
    if args.brick_mux:
        _bricks(bmc, d, f"before {step}", args)
    st["dropped"] = []
    st["stage"] = {"t0": time.time(), "before": before, "last": before, "since": "the stage's before read", "pset": {}}
    st["stage"].update(chips=chips(), aer=aer.read_aer(PCI))
    _poll_start(st, d, args, step=step)
    return []


def _after(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    if "vcore_pid" in st:  # before any capture: the sampler never reads a hung chip
        _poll_stop(st, d, "vcore")
        for ln in vcore_lines(d):
            log(ln)
    if (s := st.get("stage")) and bmc.cmd:
        t0 = (s.get("part") or [step, s["t0"]])[1]
        got, set_now = [], []
        for k, r in _regs(bmc, ("STATUS_WORD", "STATUS_INPUT")).items():
            last = s["last"].get(k, {})
            if new := _bits(r) - _bits(last):
                got.append(f"{k} {'+'.join(sorted(new))}")
                s["pset"].setdefault(k, []).append(
                    f"{step} ({_hhmm(t0)}-{_hhmm(time.time())}) set {'+'.join(sorted(new))}"
                )
            if b := _bits(r):
                set_now.append(f"{k} {'+'.join(sorted(b))}")
            s["last"][k] = {n: r[n] if r[n] is not None else last.get(n) for n in r}
        log(
            f"HSC read after {step} ({_hhmm(t0)}-{_hhmm(time.time())}), no clear: set since {s['since']}: {', '.join(got) or 'none'}; power bits set now: {', '.join(set_now) or 'none'}"
        )
        s["since"] = step
    if s and (ln := load_line(d, (s.get("part") or [step, s["t0"]])[1], step)):
        log(ln)
    _drops(bmc, st, d, step, status, args)
    _guard(st, d, step, args)
    return []


def _stage_end(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    _poll_stop(st, d)
    s, out = st.pop("stage", None), []
    if bmc.cmd and st["run"].get("regs"):
        peaks = _regs(bmc, ("READ_IOUT_PEAK", "READ_TEMP_PEAK")) if s else {}
        after = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT", "READ_VIN", "READ_IOUT"))
        before = s["before"] if s else st["run"].get("cleared") or st["run"]["regs"]  # no stage-start: the whole run
        out += _judge(step, before, after, peaks, _vin_rows(d, s["t0"]) if s else {}, st, args, s["pset"] if s else {})
    if s:
        if ln := load_line(d, s["t0"], step):
            log(ln)
        out += _power_lines(d, s, step, st) + _aer_rate(d, s["t0"], step)
        out += aer.aer_lines(s["aer"], now := aer.read_aer(PCI), (time.time() - s["t0"]) / 3600, f"during {step}")
        st["aer_last"] = now
    if args.brick_mux:
        if s and (ln := _brick_load(d, step, s["t0"])):
            log(ln)
        _bricks(bmc, d, f"after {step}", args)
    return out


def _reset(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    st.setdefault("resets", []).append(time.time())
    st["dropped"] = []
    if bmc.cmd:
        r = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT"))
        latched = (
            f"{k} {'not read' if v['STATUS_WORD'] is None else '+'.join(sorted(_bits(v))) or 'none'}"
            for k, v in r.items()
        )
        log(f"HSC latch before {step}: " + ", ".join(latched))
    have = set(chips())
    if trays := sorted({_tray(b) for b in ALL_BDFS if b not in have} - set(st.get("dumped", {}).get(step, []))):
        st.setdefault("dumped", {})[step] = trays
        _evidence(bmc, st, d, f"the {step} reset", trays, args)
    last = st.get("aer_last") or (st.get("stage") or {}).get("aer")
    st["aer_last"] = now = aer.read_aer(PCI)
    return aer.aer_lines(last, now, 0.0, f"before {step}") if last is not None else []


def run(event: str, step: str, status: str, outdir, args) -> list[str]:
    """Handle one event (see the module doc); the finding lines. Never raises for a hardware read that fails."""
    d = Path(outdir) / "captures"
    d.mkdir(parents=True, exist_ok=True)
    try:
        st = json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        st = {}
    bmc = Bmc(args.ipmitool, args.i2c_lock)
    first = "run" not in st
    out = _run_start(bmc, st, args) if first else []
    if first:  # idle, before any load or reset
        _bricks(bmc, d, "at run start", args, idle=True)
    if event == "before" and "stage" in st:
        st["stage"]["part"] = [step, time.time()]
    if event == "before" and step == "heavy":  # the heavy step's core voltage, read through it only
        _poll_start(st, d, args, "vcore")
    elif fn := dict(zip(EVENTS, (_stage_start, None, _after, _stage_end, _reset))).get(event):
        out += fn(bmc, st, d, step, status, args)
    (d / "state.json").write_text(json.dumps(st))
    return out


def sample(args) -> None:
    """One sweep for a telemetry stream: the '#route' line, a row per HSC register and brick, '#note' lines."""
    bmc = Bmc(args.ipmitool, args.i2c_lock)
    print(f"#route ok {' '.join(bmc.cmd)}" if bmc.cmd else f"#route none {bmc.why}")
    if not bmc.cmd:
        return
    for t in HSC_BUS:
        for a in HSC_ADDRS:
            for n in ("READ_VIN", "READ_IOUT", "READ_PIN", "STATUS_WORD", "STATUS_INPUT"):
                v = bmc.hsc(t, a, n)
                print(f"{time.time():.3f}\t{t}\t{a:#04x}\t0x{hsc.REG[n][0]:02X}\t{'-' if v is None else v}")
    if args.brick_mux and _brick_host(args):
        rows, notes, fails = hsc.brick_pass(bmc)
        for r in rows:
            print("\t".join([f"{time.time():.0f}", r[0], "ina238" if r[0] == "FAN" else "brick", *map(_f3, r[1:])]))
        for n in notes:
            print(f"#note {n}")
        for n in fails:  # stderr: a busy BMC's dropped reads stay out of the stream's notes
            log(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m diag.captures", description=__doc__.split("\n\n")[0])
    ap.add_argument("event", choices=[*EVENTS, "sample", "poll", "vcore"])
    ap.add_argument("args", nargs="*", help="STEP STATUS OUTDIR for an event; DIR STEP for poll; DIR for vcore")
    ap.add_argument("--watch-pgid", type=int, default=0, help=argparse.SUPPRESS)
    add_args(ap)
    a = ap.parse_args(argv)
    if a.event == "sample":
        sample(a)
    elif a.event == "poll" and len(a.args) == 2:
        poll(Path(a.args[0]), a.watch_pgid, a, a.args[1])
    elif a.event == "vcore" and len(a.args) == 1:
        vcore(Path(a.args[0]), a.watch_pgid, a)
    elif len(a.args) == 3:
        for ln in run(a.event, *a.args, a):
            print(ln, flush=True)
    else:
        ap.error("an event takes STEP STATUS OUTDIR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
