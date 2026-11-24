# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hardware captures around each stage and step: tray power, drops, hangs and PCIe errors.

``run(event, step, status, outdir, args)`` handles one event and returns its finding lines (FAIL and WARN); the other
lines go to the log and the files to ``<outdir>/captures/``. State lives in ``captures/state.json``, so the same calls
work in one process or as separate ``python3 -m diag.captures`` runs:

- the first call in an outdir records the run-start HSC latches, limits and peaks (a bit or peak already set: one
  pre-existing WARN) and, with ``--hsc-clear``, sends CLEAR_FAULTS once per HSC (none when any HSC reads OFF);
- ``stage-start``: the HSC read before the load, peaks zeroed, the idle feed table, the bricks (``--brick-mux``), the AER
  and chip baselines, and a poller (HSC VIN/IOUT, BMC power sensors, each tray's VR power-good, per-chip power, hung
  chips; ``--hang-guard`` takes a hung chip off PCIe at once, before a reader can stall a CPU core on it);
- ``before`` / ``after``: each step's HSC flags; a tray that lost chips or a hung or failed step gets its evidence
  (tray power dump, vendor CPLD dump when installed, SEL and kernel log);
- ``stage-end``: the HSC judge (new latches FAIL; low VIN, 12 V sag and peaks at the warn limit WARN; under 80% of the
  planned load a clean HSC does not count), VR power-good falls, per-chip peaks and the stage's AER errors;
- ``reset``: before a reset or re-power, the HSC latches, the dump of each tray with chips missing, and the AER errors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import aer, hsc
from .common import SUITE_DIR, log, resolve_optional_tool, run_with_timeout
from .galaxy import expected_bdf, parse_bdf_physical
from .hsc import HSC_ADDRS, HSC_BUS, INPUT_BITS, IOUT_LSB, VIN_LSB, WORD_BITS, Bmc, tray_bdfs

EVENTS = ("stage-start", "before", "after", "stage-end", "reset", "run")
LOAD_JUDGED_PCT = 80  # under this share of the planned load a clean HSC proves nothing
VIN_MIN_V = 11.0  # a loaded VIN under this: WARN
SAG_MOHM, SAG_INFO_MOHM = 13.0, 11.0  # 12 V sag (idle VIN - VIN) / IOUT: WARN over, said over
SAG_MIN_A, IDLE_MAX_A = 40, 30  # sag samples at or above; idle VIN from samples at or below
PEAK_PCT = 20  # a feed's peak this far over the median of the others: WARN
OC_LIMIT_A = (50, 1000)  # a plausible IOUT_OC limit; outside: not read
POWER_S = 2  # power poll period
AER_S = 60  # corrected AER rate poll period
RESET_MASK_S = 40  # VR power-good falls this near a reset are the reset's
COLLAPSE_W, HOLD_W = 50, 600  # one tray at or under COLLAPSE_W while every other holds HOLD_W: its 12 V collapsed
CPLD_DUMP_MAX = 4
PCI = Path("/sys/bus/pci/devices")


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("hardware captures")
    g.add_argument(
        "--hsc-clear", action="store_true", help="CLEAR_FAULTS once per tray HSC per run (none when any reads OFF)"
    )
    g.add_argument(
        "--brick-mux", action="store_true", help="read the power bricks: select their mux channel, read, restore"
    )
    g.add_argument(
        "--i2c-lock", help="lock file every in-band BMC I2C transaction takes (default /tmp/hc-bmc-<uid>.lock)"
    )
    g.add_argument("--site-data", help="site rows: 'brick_mux <host|*> <why>'")
    g.add_argument("--load-pct", type=float, default=100.0, help="share of the planned load the stage reached")
    g.add_argument("--hang-guard", action="store_true", help="take a hung chip off PCIe (root only)")
    g.add_argument("--cpld-dump", help="the vendor CPLD dump script (default: cpld-dump.sh on PATH)")
    g.add_argument("--ipmitool", help="the ipmitool command as is (default ipmitool, or sudo -n ipmitool)")


# ── state, chips, files ──────────────────────────────────────────────────────


def _key(t: int, a: int) -> str:
    return f"{t}:{a:#04x}"


def _where(k: str) -> str:
    t, a = k.split(":")
    return f"tray {t} ({tray_bdfs(int(t))}) HSC {a}"


def _num(raw, lsb: float, unit: str) -> str:
    return "not read" if raw is None else f"{raw * lsb:.2f} {unit}"


def _hx(x) -> str:
    return "not read" if x is None else f"{x:#04x}"


def _hhmm(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _regs(bmc: Bmc, names, trays=HSC_BUS) -> dict[str, dict]:
    return {_key(*k): v for k, v in hsc.read_regs(bmc, names, trays).items()} if bmc.cmd else {}


def chips() -> list[str]:
    return [d.name for d in aer.tt_devices(PCI)]


def hung(bdf: str) -> bool:
    """Config space reads all ones: the chip's link is gone and an MMIO read of it would stall the CPU."""
    try:
        return (PCI / bdf / "config").read_bytes()[:4] == b"\xff" * 4
    except OSError:
        return False


def _tray(bdf: str) -> int | None:
    return (parse_bdf_physical(bdf) or (None,))[0]


def _site(args) -> list[list[str]]:
    try:
        rows = Path(args.site_data).read_text().splitlines() if args.site_data else []
    except OSError:
        rows = []
    return [r.split() for r in rows if r.strip() and not r.lstrip().startswith("#")]


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _alive(pid: int) -> bool:
    try:
        if os.waitpid(pid, os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _save_cmd(path: Path, argv: list[str], what: str, timeout_s: float = 60) -> str:
    try:
        rc, out, err, _ = run_with_timeout(argv, timeout_s)
    except OSError as e:
        return f"{what} not saved: {e}"
    path.write_text(out + err)
    return f"{what} saved: {path}" + (f" (rc {rc})" if rc else "")


# ── HSC ──────────────────────────────────────────────────────────────────────


def _bits(sw, si) -> set[str]:
    return ({n for b, n in WORD_BITS.items() if sw & b} if sw is not None else set()) | (
        {n for b, n in INPUT_BITS.items() if si & b} if si is not None else set()
    )


def _plaus(x) -> float | None:
    return float(x) if x is not None and OC_LIMIT_A[0] <= x <= OC_LIMIT_A[1] else None


def _run_start(bmc: Bmc, st: dict, args) -> list[str]:
    st["run"] = {"t": time.time()}
    if not bmc.cmd:
        log(f"HSC check not run: {bmc.why}; tray 12 V under-voltage latches and min VIN not judged")
        return []
    names = ("STATUS_WORD", "STATUS_INPUT", "FAULT_RECORD", "FAULT_RECORD_LAST", "STATUS_IOUT", "READ_VOUT")
    regs = _regs(bmc, (*names, "IOUT_OC_FAULT_LIMIT", "IOUT_OC_WARN_LIMIT", *hsc.VIN_LIMITS, "READ_IOUT_PEAK"))
    st["run"]["regs"] = regs
    if not any(r["STATUS_WORD"] is not None for r in regs.values()):
        log("HSC check not run: no HSC answered; tray 12 V under-voltage latches and min VIN not judged")
        return []
    pre = []
    for k, r in regs.items():
        sw, si = r["STATUS_WORD"], r["STATUS_INPUT"]
        log(f"HSC run start {_where(k)}: " + ", ".join(f"{n} {_hx(r[n])}" for n in names))
        log(f"HSC VIN limits {_where(k)}: " + ", ".join(f"{n} {hsc.vin_limit(n, r[n])}" for n in hsc.VIN_LIMITS))
        if b := _bits(sw, si):
            pre.append(f"{_where(k)}: {'+'.join(sorted(b))} (STATUS_WORD {_hx(sw)}, STATUS_INPUT {_hx(si)})")
        if sw is not None and sw & hsc.CML:
            pre.append(f"{_where(k)}: comm fault (CML), not power")
        p, w = r["READ_IOUT_PEAK"], _plaus(r["IOUT_OC_WARN_LIMIT"])
        if p is not None and w and p * IOUT_LSB >= w:
            pre.append(f"{_where(k)}: IOUT peak {p * IOUT_LSB:.1f} A at or above its own IOUT_OC_WARN_LIMIT {w:.0f} A")
    out = [f"HSC WARN host: {'; '.join(pre)}; latched before the run: pre-existing, not from this run"] if pre else []
    if args.hsc_clear:
        off = [k for k, r in regs.items() if r["STATUS_WORD"] is not None and r["STATUS_WORD"] & 0x40]
        if off:
            log(
                f"HSC latch clear skipped: HSC OFF ({', '.join(map(_where, off))}); no CLEAR_FAULTS sent to any HSC this run"
            )
        else:
            sent = [k for k, r in regs.items() if r["STATUS_WORD"] is not None and bmc.clear_faults(*_tk(k))]
            st["run"]["cleared"] = after = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT"))
            left = [
                f"{k} {'+'.join(sorted(b))}"
                for k, r in after.items()
                if (b := _bits(r["STATUS_WORD"], r["STATUS_INPUT"]))
            ]
            log(f"HSC latch: CLEAR_FAULTS sent once to {len(sent)} HSC(s); after it: {', '.join(left) or 'all clear'}")
    return out


def _tk(k: str) -> tuple[int, int]:
    t, a = k.split(":")
    return int(t), int(a, 16)


def _vin_rows(d: Path, t0: float) -> dict[str, list[tuple[float, float | None]]]:
    """The poller's samples since t0 per HSC: (VIN V, IOUT A or None), in range only (6-15 V, IOUT up to 250 A)."""
    out: dict[str, list] = {}
    try:
        lines = (d / "vin.tsv").read_text().splitlines()
    except OSError:
        return out
    for ln in lines:
        f = ln.split("\t")
        if len(f) != 5 or float(f[0]) < t0 or not f[3].isdigit():
            continue
        v, i = int(f[3]) * VIN_LSB, int(f[4]) * IOUT_LSB if f[4].isdigit() else None
        if 6 <= v <= 15 and int(f[3]) != 0xFFFF and (i is None or i <= 250):
            out.setdefault(f"{f[1]}:{f[2]}", []).append((v, i))
    return out


def _judge(step: str, before: dict, after: dict, peaks: dict, samples: dict, st: dict, args, pset: dict) -> list[str]:
    out, info, named, sag = [], [], set(st.setdefault("named", [])), {}
    lims = st["run"].get("regs", {})
    for k, a in after.items():
        b = before.get(k) or {}
        sw0, si0, sw1, si1 = b.get("STATUS_WORD"), b.get("STATUS_INPUT"), a["STATUS_WORD"], a["STATUS_INPUT"]
        s = samples.get(k, [])
        vins = [v for v, _ in s] + [
            x * VIN_LSB for x in (b.get("READ_VIN"), a["READ_VIN"]) if x and 6 <= x * VIN_LSB <= 15
        ]
        uv = [n for bit, n in INPUT_BITS.items() if None not in (si0, si1) and si1 & bit and not si0 & bit]
        wd = [n for bit, n in WORD_BITS.items() if None not in (sw0, sw1) and sw1 & bit and not sw0 & bit]
        uv, wd = [n for n in uv if f"{k}:{n}" not in named], [n for n in wd if f"{k}:{n}" not in named]
        if si0 is not None and si0 & 0x30:
            info.append(
                f"HSC {_where(k)} latched before load: a new dip on this tray cannot be seen until a power cycle clears it"
            )
        regs = (
            f"STATUS_INPUT {si0:#04x} -> {si1:#04x}, STATUS_WORD {sw0:#06x} -> {sw1:#06x}"
            if None not in (sw0, si0, sw1, si1)
            else ""
        )
        why = [f"{'+'.join(uv)} latched during {step} (clear before; {regs})"] if uv else []
        if "VIN_UV_WARN" in uv:
            lim = hsc.vin_limit("VIN_UV_WARN_LIMIT", lims.get(k, {}).get("VIN_UV_WARN_LIMIT"))
            low = f"poll min {min(v for v, _ in s):.2f} V over {len(s)} sample(s)" if s else "no poll sample"
            why.append(f"VIN fell below the UV warn limit {lim} (58h) at least once, depth unknown ({low})")
        if wd:
            why.append(f"{'+'.join(wd)} set during {step} ({regs})")
        if why:
            parts = "; per-part reads: " + ", then ".join(pset[k]) if pset.get(k) else ""
            out.append(f"HSC FAIL {_where(k)}: {'; '.join(why)}{parts}")
            named |= {f"{k}:{n}" for n in uv + wd}
        elif vins and min(vins) < VIN_MIN_V:
            out.append(
                f"HSC WARN {_where(k)}: min VIN {min(vins):.2f} V during {step}, below {VIN_MIN_V} V ({len(s)} sample(s)); WARN only, not a finding"
            )
        idle = [v for v, i in s if i is not None and 2 < i <= IDLE_MAX_A]
        ref = (
            statistics.median(idle)
            if len(idle) >= 3
            else (b["READ_VIN"] * VIN_LSB if b.get("READ_VIN") is not None else None)
        )
        rs = [((ref - v) / i * 1000, v, i) for v, i in s if ref is not None and i is not None and i >= SAG_MIN_A]
        if rs:
            r, v, i = max(rs)
            sag[k] = r
            line = f"{_where(k)}: 12 V sag {r:.1f} mOhm during {step} (VIN {v:.2f} V at IOUT {i:.1f} A, idle VIN {ref:.2f} V)"
            if r > SAG_MOHM:
                out.append(f"HSC WARN {line}, over {SAG_MOHM} mOhm; WARN only, not a finding")
            elif r > SAG_INFO_MOHM:
                info.append(f"HSC sag information {line}, over {SAG_INFO_MOHM} mOhm: information only, not a warn")
    pk = {k: r["READ_IOUT_PEAK"] * IOUT_LSB for k, r in peaks.items() if r.get("READ_IOUT_PEAK") is not None}
    for k, p in pk.items():
        w, f = (_plaus(lims.get(k, {}).get(n)) for n in ("IOUT_OC_WARN_LIMIT", "IOUT_OC_FAULT_LIMIT"))
        if w and p >= w:
            trip = (
                f"trip IOUT_OC_FAULT_LIMIT {f:.0f} A (46h), margin {f - p:.1f} A"
                if f
                else "IOUT_OC_FAULT_LIMIT (46h) not read"
            )
            out.append(
                f"HSC WARN {_where(k)}: IOUT peak {p:.1f} A during {step}, at or above its own IOUT_OC_WARN_LIMIT {w:.0f} A (4Ah), {trip}; WARN only, not a tray verdict"
            )
        others = [x for j, x in pk.items() if j != k]
        if len(others) >= 3 and (m := statistics.median(others)) > 0 and p > m * (1 + PEAK_PCT / 100):
            out.append(
                f"HSC WARN {_where(k)}: IOUT peak {p:.1f} A during {step}, {(p / m - 1) * 100:.0f}% above the median {m:.1f} A of the other {len(others)} feeds; WARN only, not a finding"
            )
    st["named"] = sorted(named)
    uvl = [k for k, a in after.items() if a["STATUS_INPUT"] is not None and a["STATUS_INPUT"] & 0x30]
    judged = (
        "UV latched: " + (", ".join(uvl) or "none")
        if uvl or args.load_pct >= LOAD_JUDGED_PCT
        else f"UV not judged: load not reached ({args.load_pct:.0f}% of the planned load)"
    )
    vmin = ", ".join(f"{k} {min(v for v, _ in s):.2f}" for k, s in samples.items() if s)
    info.append(
        f"HSC after {step}: {judged}; poll min VIN V {vmin or 'none'}; sag R max mOhm "
        + (", ".join(f"{k} {r:.1f}" for k, r in sag.items()) or "none")
    )
    for t in HSC_BUS:  # each tray's HSCs summed per poll round
        per = [[i for _, i in samples.get(_key(t, a), []) if i is not None] for a in HSC_ADDRS]
        tot = sorted(sum(x) for x in zip(*per))
        if tot:
            info.append(
                f"HSC load tray {t} ({tray_bdfs(t)}) during {step}: IOUT mean {statistics.mean(tot):.1f} A, p90 {tot[int(0.9 * (len(tot) - 1))]:.1f} A ({len(tot)} poll(s))"
            )
    for t in HSC_BUS:  # the tray power line: a tray whose HSC turned off
        offs = [ln for ln in out if ln.startswith(f"HSC FAIL tray {t} ") and "OFF" in ln.split(": ", 1)[1]]
        if offs:
            ranked = sorted(sag.values(), reverse=True)
            items = []
            for ln in offs:
                k = f"{t}:{ln.split(' HSC ')[1][:4]}"
                bits = " + ".join(sorted({"OFF", *(n for n in (*INPUT_BITS.values(), "IOUT_OC_FAULT") if n in ln)}))
                items.append(
                    f"HSC {k[2:]} {bits}"
                    + (
                        f", sag {sag[k]:.1f} mOhm {'top' if ranked.index(sag[k]) == 0 else f'rank {ranked.index(sag[k]) + 1}'} of {len(ranked)}"
                        if k in sag
                        else ""
                    )
                )
            st.setdefault("tray_power", {})[str(t)] = "; ".join(items)
            info.append(f"HSC power tray {t}: {'; '.join(items)}")
    for ln in info:
        log(ln)
    return out


# ── bricks ───────────────────────────────────────────────────────────────────


def _bricks(bmc: Bmc, d: Path, when: str, args) -> None:
    host = socket.gethostname().split(".")[0]
    if not any(r[0] == "brick_mux" and len(r) > 1 and r[1] in (host, "*") for r in _site(args)):
        log(f"WARNING brick reads skipped: no brick_mux row for {host} in --site-data {args.site_data or '(none)'}")
        return
    if not bmc.cmd:
        log(f"brick reads skipped: {bmc.why}")
        return
    rows, notes = hsc.brick_pass(bmc)
    f = lambda x, u: "-" if x is None else f"{x:.3f}{u}"  # noqa: E731
    with open(d / "brick.tsv", "a") as fh:
        for r in rows:
            fh.write("\t".join([f"{time.time():.0f}", when, r[0], *(f(x, "") for x in r[1:])]) + "\n")
    log(
        f"bricks and the 54 V rail {when}: "
        + ", ".join(f"{r[0]} {f(r[1], ' V')} {f(r[2], ' A')}" for r in rows)
        + "".join(f"; {n}" for n in notes)
    )


# ── drops, hangs, evidence ───────────────────────────────────────────────────


def _dump_tray(bmc: Bmc, d: Path, t: int, what: str) -> str:
    """The tray's power state read before a reset clears it: both HSCs and the CPLD's alive, thermal-trip and VR
    power-good bytes. One summary line; the registers go to a file."""
    f = d / f"drop_{time.strftime('%Y%m%dT%H%M%S')}_tray{t}.txt"
    if not bmc.cmd:
        return f"tray {t} ({tray_bdfs(t)}) power state not read: {bmc.why}"
    regs = _regs(bmc, [n for n in hsc.REG if "PEAK" not in n and "OC_" not in n and n != "READ_PIN"], trays=(t,))
    cp = {n: bmc.cpld(t, r) for n, r in (("alive", 0x1E), ("therm_trip", 0x2D), ("vr_pg", 0x19))}
    lines = [f"tray {t} ({tray_bdfs(t)}) power state at {what}, read only"]
    lines += [f"cpld {n} {'not read' if v is None else f'{v:#04x}'}" for n, v in cp.items()]
    lines += [f"hsc {k[2:]} {n} {'not read' if v is None else hex(v)}" for k, r in regs.items() for n, v in r.items()]
    f.write_text("\n".join(lines) + "\n")
    hs = "; ".join(
        f"HSC {k[2:]} "
        + (
            "not read"
            if r["STATUS_WORD"] is None
            else f"STATUS_WORD {r['STATUS_WORD']:#06x} VIN {(r['READ_VIN'] or 0) * VIN_LSB:.2f} V IOUT {(r['READ_IOUT'] or 0) * IOUT_LSB:.1f} A"
        )
        for k, r in regs.items()
    )
    return (
        f"tray {t} ({tray_bdfs(t)}) power state at {what}: CPLD "
        + " ".join(f"{n} {'-' if v is None else f'{v:#04x}'}" for n, v in cp.items())
        + f"; {hs} (dump {f})"
    )


def _sudo(argv: list[str]) -> list[str] | None:
    """argv as root, or under sudo -n when sudo -n -l lists it exactly; None when neither."""
    return argv if os.geteuid() == 0 else ["sudo", "-n", *argv] if hsc.sudo_listed(argv) else None


def _cpld_dump(bmc: Bmc, st: dict, d: Path, trays: list[int], args) -> str:
    """The vendor CPLD dump, once per drop event (a new set of trays), at most CPLD_DUMP_MAX per run. It selects CPLD
    debug indexes, so each tray CPLD's index (0x50) is read before and written back after."""
    tool, why = resolve_optional_tool(args.cpld_dump, "cpld-dump.sh", "--cpld-dump", "not installed")
    if not tool:
        return f"CPLD dump: {why}"
    if trays == st.get("cpld_open") or st.get("cpld_n", 0) >= CPLD_DUMP_MAX:
        return f"CPLD dump of tray(s) {trays}: not run (same event, or {CPLD_DUMP_MAX} dumps this run)"
    st["cpld_open"], st["cpld_n"] = trays, st.get("cpld_n", 0) + 1
    runs = {m: _sudo(["timeout", "-s", "KILL", "90", "bash", tool, m]) for m in ("DIAG", "PSU")}
    if None in runs.values() or not bmc.cmd:
        return (
            f"CPLD dump not run: no root, and sudo -n -l does not list 'timeout -s KILL 90 bash {tool} DIAG|PSU'"
            if bmc.cmd
            else f"CPLD dump not run: {bmc.why}"
        )
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
    lines = [_dump_tray(bmc, d, t, what) for t in trays]
    if trays:
        lines.append(_cpld_dump(bmc, st, d, trays, args))
    if not bmc.cmd:
        lines.append(f"SEL not saved: {bmc.why}")
    elif bmc.listed("sel", "elist"):
        lines.append(_save_cmd(d / f"sel_{ts}.txt", [*bmc.cmd, "sel", "elist"], "SEL"))
    else:
        lines.append("SEL not saved: no root, and sudo -n -l does not list ipmitool sel elist")
    kern = _sudo(["journalctl", "-k", "-b", "--no-pager"])
    lines.append(
        _save_cmd(d / f"kern_{ts}.txt", kern, "kernel log")
        if kern
        else "kernel log not saved: no root, and sudo -n -l does not list journalctl -k -b --no-pager"
    )
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


def _guard(st: dict, d: Path, step: str, status: str, args) -> None:
    """The hung-chip guard: a chip whose config space reads all ones stalls any CPU that reads its MMIO (a host
    watchdog machine check follows within minutes), so with --hang-guard it is taken off PCIe. The poller does this as
    soon as it sees one; here the chips it saw and any found now are named."""
    seen = {}
    try:
        for ln in (d / "hung.tsv").read_text().splitlines():
            t, b, how = ln.split("\t", 2)
            seen.setdefault(b, (float(t), how))
    except (OSError, ValueError):
        pass
    for b in (st.get("stage") or {}).get("chips", []):
        if b not in seen and hung(b):
            seen[b] = (time.time(), _remove(b, args))
    for b, (t, how) in seen.items():
        log(
            f"hang guard: {b} (tray {_tray(b)}) config space reads ffffffff, first seen {_hhmm(t)} ({time.time() - t:.0f} s before the end of {step}); {how}"
        )


def _drops(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> None:
    """Chips gone since the stage started (outside a stage: of the 32): each tray's evidence, once until the next
    stage or reset; a hung, failed or dropped step's SEL and kernel log too."""
    base = st["stage"]["chips"] if "stage" in st else [expected_bdf(t, a) for t in HSC_BUS for a in range(1, 9)]
    gone = sorted(set(base) - set(chips()))
    trays = sorted({_tray(b) for b in gone} - set(st.get("dropped", [])))
    if trays:
        st["dropped"] = sorted({*st.get("dropped", []), *trays})
        log(
            f"chips lost during {step}: {', '.join(b.split(':', 1)[1] for b in gone)} (tray {', '.join(map(str, trays))})"
        )
    if trays or status.lower() in ("hung", "hang", "fail", "drop", "lost"):
        _evidence(bmc, st, d, f"the end of {step}", trays, args)


# ── poller ───────────────────────────────────────────────────────────────────


def _hwmon(bdf: str) -> tuple[float | None, float | None]:
    out = []
    for f, scale in (("power1_input", 1e-6), ("curr1_input", 1e-3)):
        try:
            out.append(int(next((PCI / bdf / "hwmon").glob(f"hwmon*/{f}")).read_text()) * scale)
        except (OSError, StopIteration, ValueError):
            out.append(None)
    return tuple(out)


class _AerRate:
    """Each root port's corrected AER errors a minute, from sysfs and the kernel log, the larger (a dropped chip's port
    is still read): one aer_min.tsv row per port and sample."""

    def __init__(self):
        self.ports, self.kern = aer.root_ports(PCI), _sudo(["dmesg"]) or ["dmesg"]
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
            f.write(
                f"{now:.0f}\t{p.name}\t{','.join(x.name for x in cs)}\t{rise}\t{rise * 60 / max(now - self.t, 60):.0f}\n"
            )
            self.cor[p] = c
        self.t = now


def _aer_rate(d: Path, t0: float, step: str) -> list[str]:
    """A root port whose corrected AER errors rose over RATE_PER_MIN in any minute of the stage: a weak PCIe link."""
    seen: dict[str, list] = {}
    try:
        rows = [r.split("\t") for r in (d / "aer_min.tsv").read_text().splitlines()]
    except OSError:
        rows = []
    for ts, port, cs, rise, pm in (r for r in rows if len(r) == 5 and float(r[0]) >= t0):
        s = seen.setdefault(port, [cs, 0, 0, 0, 0])  # chips, minutes, minutes over, peak, sum
        s[1:] = [s[1] + 1, s[2] + (int(pm) > aer.RATE_PER_MIN), max(s[3], int(pm)), s[4] + int(rise)]
    return [
        f"AER WARN weak PCIe link {', '.join(f'{b[5:]} (tray {_tray(b)})' for b in cs.split(','))}: root port {port} "
        f"corrected AER peak {peak}/min (over {aer.RATE_PER_MIN}/min in {over} of {n} min, +{tot} in all) during {step}"
        for port, (cs, n, over, peak, tot) in seen.items()
        if over
    ]


def poll(d: Path, watch: int, args) -> None:
    """Until stage-end's stop file, or the caller's process group *watch* is gone: each HSC's VIN (IOUT every other
    sweep) to vin.tsv; every POWER_S the BMC power sensors and each tray's VR power-good byte to power.tsv, each chip's
    power and current to chips.tsv, a newly hung chip to hung.tsv (taken off PCIe with --hang-guard); every AER_S each
    root port's corrected AER rate to aer_min.tsv."""
    bmc, stop, bdfs, sweep, nxt = Bmc(args.ipmitool, args.i2c_lock), d / "poll.stop", chips(), 0, 0.0
    rate, nxt_aer = _AerRate(), time.time() + AER_S
    with open(d / "vin.tsv", "a") as fv, open(d / "power.tsv", "a") as fp, open(d / "chips.tsv", "a") as fc, open(
        d / "hung.tsv", "a"
    ) as fh, open(d / "aer_min.tsv", "a") as fa:
        while not stop.exists() and _group_alive(watch):
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
                    pg = [(lambda v: "ERR" if v is None else f"{v:02x}")(bmc.cpld(t, 0x19)) for t in HSC_BUS]
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
                for f in (fv, fp, fc, fh, fa):
                    f.flush()
            time.sleep(0.05)
        rate.sample(fa)  # the stage's last minute


def _poll_start(st: dict, d: Path, args) -> None:
    (d / "poll.stop").unlink(missing_ok=True)
    flags = [
        x for k in ("ipmitool", "i2c_lock") if getattr(args, k) for x in (f"--{k.replace('_', '-')}", getattr(args, k))
    ]
    argv = [sys.executable, "-m", "diag.captures", "poll", str(d), "--watch-pgid", str(os.getpgrp()), *flags]
    with open(d / "poll.err", "a") as err:
        p = subprocess.Popen(
            [*argv, *(["--hang-guard"] if args.hang_guard else [])],
            cwd=SUITE_DIR,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
    st["poll_pid"] = p.pid


def _poll_stop(st: dict, d: Path) -> None:
    pid = st.pop("poll_pid", None)
    if not pid:
        return
    (d / "poll.stop").touch()
    for _ in range(150):  # one HSC sweep at most
        if not _alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _power_lines(d: Path, s: dict, step: str, st: dict) -> list[str]:
    """VR power-good falls (a tray CPLD byte not ff on two polls in a row) and 12 V collapses (one tray at or under
    COLLAPSE_W while every other holds HOLD_W) during the stage, away from resets; each chip's peak power and current.
    """
    out, t0, resets = [], s["t0"], st.get("resets", [])
    try:
        rows = [r.split() for r in (d / "power.tsv").read_text().splitlines()]
    except OSError:
        rows = []
    rows = [
        r
        for r in rows
        if len(r) >= 10 and float(r[0]) >= t0 and all(abs(float(r[0]) - x) > RESET_MASK_S for x in resets)
    ]
    for t in HSC_BUS:
        run, fell = [], None
        for r in rows:
            v = r[5 + t].lower()
            run = [] if v == "ff" else run + [(r[0], v)] if len(v) == 2 else run
            if len(run) >= 2 and not fell:
                fell = run[0]
        if fell:
            out.append(
                f"12V FAIL tray {t} ({tray_bdfs(t)}): vr_pg {fell[1]} from {_hhmm(float(fell[0]))} during {step} (not ff on two polls in a row)"
            )
        w = [(r[0], [int(x) if x.isdigit() else None for x in r[1:5]]) for r in rows]
        low = next(
            (
                (ts, p[t - 1])
                for ts, p in w
                if None not in p and p[t - 1] <= COLLAPSE_W and all(x >= HOLD_W for j, x in enumerate(p) if j != t - 1)
            ),
            None,
        )
        if low:
            out.append(
                f"12V FAIL tray {t} ({tray_bdfs(t)}): Power_UBB{t - 1} {low[1]} W at {_hhmm(float(low[0]))} while every other tray held {HOLD_W} W or more during {step}"
            )
    peak: dict[str, list[float]] = {}
    try:
        for ln in (d / "chips.tsv").read_text().splitlines():
            ts, b, w, a = ln.split("\t")
            if float(ts) >= t0:
                p = peak.setdefault(b, [0.0, 0.0])
                p[0], p[1] = max(p[0], float(w)), max(p[1], float(a))
    except (OSError, ValueError):
        pass
    if peak:
        f = d / f"chip_peak_{step.replace(' ', '_').replace('/', '_')}.tsv"
        f.write_text("".join(f"{b}\t{_tray(b)}\t{w:.1f}\t{a:.1f}\n" for b, (w, a) in sorted(peak.items())))
        bw, ba = max(peak, key=lambda b: peak[b][0]), max(peak, key=lambda b: peak[b][1])
        log(
            f"peak per chip during {step}: {peak[bw][0]:.1f} W ({bw}, tray {_tray(bw)}), {peak[ba][1]:.1f} A ({ba}, tray {_tray(ba)}); per chip in {f}"
        )
    return out


# ── events ───────────────────────────────────────────────────────────────────


def _stage_start(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    _poll_stop(st, d)  # a stage whose stage-end never came
    before = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT", "READ_VIN", "READ_IOUT"))
    _regs(bmc, ("READ_IOUT_PEAK", "READ_TEMP_PEAK"))  # read-clear: the read after the stage is the stage's own peak
    vin = {k: r["READ_VIN"] * VIN_LSB for k, r in before.items() if r["READ_VIN"] is not None}
    for k, r in before.items():
        v, i = r["READ_VIN"], r["READ_IOUT"]
        log(f"HSC idle {_where(k)} before {step}: VIN {_num(v, VIN_LSB, 'V')}, IOUT {_num(i, IOUT_LSB, 'A')}")
    if vin:
        lo, hi = min(vin, key=vin.get), max(vin, key=vin.get)
        log(
            f"HSC idle box before {step}: VIN median {statistics.median(vin.values()):.3f} V, min {vin[lo]:.3f} V ({lo}), max {vin[hi]:.3f} V ({hi})"
        )
    if args.brick_mux:
        _bricks(bmc, d, f"before {step}", args)
    st["dropped"] = []
    st["stage"] = {
        "name": step,
        "t0": time.time(),
        "before": before,
        "last": before,
        "since": "the stage's before read",
        "pset": {},
        "chips": chips(),
        "aer": aer.read_aer(PCI),
    }
    _poll_start(st, d, args)
    return []


def _before(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    if "stage" in st:
        st["stage"]["part"] = [step, time.time()]
    return []


def _after(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    s = st.get("stage")
    if s and bmc.cmd:
        now = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT"))
        t0 = (s.get("part") or [step, s["t0"]])[1]
        got, set_now = [], []
        for k, r in now.items():
            last = s["last"].get(k, {})
            new = _bits(r["STATUS_WORD"], r["STATUS_INPUT"]) - _bits(last.get("STATUS_WORD"), last.get("STATUS_INPUT"))
            if new:
                got.append(f"{k} {'+'.join(sorted(new))}")
                s["pset"].setdefault(k, []).append(
                    f"{step} ({_hhmm(t0)}-{_hhmm(time.time())}) set {'+'.join(sorted(new))}"
                )
            if b := _bits(r["STATUS_WORD"], r["STATUS_INPUT"]):
                set_now.append(f"{k} {'+'.join(sorted(b))}")
            s["last"][k] = {n: r[n] if r[n] is not None else last.get(n) for n in r}
        log(
            f"HSC read after {step} ({_hhmm(t0)}-{_hhmm(time.time())}), no clear: set since {s['since']}: {', '.join(got) or 'none'}; power bits set now: {', '.join(set_now) or 'none'}"
        )
        s["since"] = step
    _drops(bmc, st, d, step, status, args)
    _guard(st, d, step, status, args)
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
        out += _power_lines(d, s, step, st) + _aer_rate(d, s["t0"], step)
        out += aer.aer_lines(s["aer"], now := aer.read_aer(PCI), (time.time() - s["t0"]) / 3600, f"during {step}")
        st["aer_last"] = now
        if args.brick_mux:
            _bricks(bmc, d, f"after {step}", args)
    return out


def _reset(bmc: Bmc, st: dict, d: Path, step: str, status: str, args) -> list[str]:
    st.setdefault("resets", []).append(time.time())
    st["dropped"] = []
    if bmc.cmd:
        r = _regs(bmc, ("STATUS_WORD", "STATUS_INPUT"))
        log(
            f"HSC latch before {step}: "
            + ", ".join(
                f"{k} {'not read' if v['STATUS_WORD'] is None else '+'.join(sorted(_bits(v['STATUS_WORD'], v['STATUS_INPUT']))) or 'none'}"
                for k, v in r.items()
            )
        )
    have = set(chips())
    trays = sorted(
        {t for t in HSC_BUS for a in range(1, 9) if expected_bdf(t, a) not in have}
        - set(st.get("dumped", {}).get(step, []))
    )
    if trays:
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
    bmc, out = Bmc(args.ipmitool, args.i2c_lock), []
    if "run" not in st:
        out += _run_start(bmc, st, args)
    fn = {
        "stage-start": _stage_start,
        "before": _before,
        "after": _after,
        "stage-end": _stage_end,
        "reset": _reset,
    }.get(event)
    if fn:
        out += fn(bmc, st, d, step, status, args)
    (d / "state.json").write_text(json.dumps(st))
    return out


def sample(args) -> None:
    """One sweep for a telemetry stream: '#route ok <cmd>' or '#route none <why>', then a row per HSC register and,
    with --brick-mux on an allowed host, a row per brick; '#note' lines for what did not read."""
    bmc = Bmc(args.ipmitool, args.i2c_lock)
    print(f"#route ok {' '.join(bmc.cmd)}" if bmc.cmd else f"#route none {bmc.why}")
    if not bmc.cmd:
        return
    for t in HSC_BUS:
        for a in HSC_ADDRS:
            for n in ("READ_VIN", "READ_IOUT", "READ_PIN", "STATUS_WORD", "STATUS_INPUT"):
                v = bmc.hsc(t, a, n)
                print(f"{time.time():.3f}\t{t}\t{a:#04x}\t0x{hsc.REG[n][0]:02X}\t{'-' if v is None else v}")
    host = socket.gethostname().split(".")[0]
    if args.brick_mux and any(r[0] == "brick_mux" and len(r) > 1 and r[1] in (host, "*") for r in _site(args)):
        rows, notes = hsc.brick_pass(bmc)
        for r in rows:
            f = ["-" if x is None else f"{x:.3f}" for x in r[1:]]
            print("\t".join([f"{time.time():.0f}", r[0], "ina238" if r[0] == "FAN" else "brick", *f]))
        for n in notes:
            print(f"#note {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m diag.captures", description=__doc__.split("\n\n")[0])
    ap.add_argument("event", choices=[*EVENTS, "sample", "poll"])
    ap.add_argument("args", nargs="*", help="STEP STATUS OUTDIR for an event; DIR for poll")
    ap.add_argument("--watch-pgid", type=int, default=0, help=argparse.SUPPRESS)
    add_args(ap)
    a = ap.parse_args(argv)
    if a.event == "sample":
        sample(a)
    elif a.event == "poll" and len(a.args) == 1:
        poll(Path(a.args[0]), a.watch_pgid, a)
    elif len(a.args) == 3:
        for ln in run(a.event, *a.args, a):
            print(ln, flush=True)
    else:
        ap.error("an event takes STEP STATUS OUTDIR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
