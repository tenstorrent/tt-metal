#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""load_line.py: the load-strength line of one ub0_repro.py burst.

Usage: load_line.py <power.txt> <result.json> <label> [<c8_telemetry.txt> <c8_limits.txt> [<telem.csv>]]
       load_line.py chunks <rows.jsonl> <c8_telemetry.txt> <c8_limits.txt> <label> [<out.json>]
       load_line.py chips <rows.jsonl | result.json> <c8_telemetry.txt> <c8_limits.txt> <label>
       load_line.py target <rows.jsonl | result.json> <c8_telemetry.txt> <c8_limits.txt> <label> [<fault_times.tsv>]   (LOAD LOW; fault cut-off)
power.txt: fleet.sh's poll rows "<epoch> <tray 1-4 W> <HSC W> <vr_pg tray 1-4>"; result.json: ub0_repro.py's
--result (burst t0, seconds, tflops_per_chip, op, fallback). Prints one line:
  LOAD <label>: trays' sum idle <i> kW, peak <p> kW in the first 10 s (the step), burst mean <m> kW (<n> polls),
  tray 1-4 peak a/b/c/d W, <x> TFLOP/s per chip (matmul FLOPs over the burst time, gaps included), op <op that ran>[ (fallback: <why>)]; <power limit>
Power limit (does every load hit the chips' power limit?): C8's per-chip sysfs rows
"<epoch> <bdf> <temp mC> <power uW> <aiclk> <current mA> <vcore mV>" and its limits "<bdf> <power1_max uW>
<curr1_max mA>" (TDP, TDC). A chip is at its limit when its mean power over the stage (its first SKIP_S s, the ramp,
left out) is at least AT_LIMIT of TDP, or its mean current of TDC. The count goes in the line,
with each chip not at it by BDF and tray 1-4, and into result.json's "power_limit". The chunks form prints one such
line per sustained chunk (chunk_sched.py's rows) and writes them to <out.json>.
With <telem.csv> (ub0_repro.py --telem: one row per chip per 1 s sweep, columns utc,dev,pci,tray,aiclk_mhz,power_w,..)
the line ends with the per-tray swing (the bus-bar stage): each tray's chip-sum kW, min-max over the 25 s
before the burst and the burst, and the BMC tray sensors' min-max (they cap at 2550 W, so the chip sums are the swing).
The chips form (is the load ACTUALLY the hardware's max on every chip?) prints one line per
load stage with each chip's mean W, A and MHz next to its own power and current limit, the stage's TFLOP/s per chip, and
which limit each chip hits (power, current, clock throttle, none); then a "LOAD WARN" line per chip whose median power
over the stage's load polls is under MED_WARN of the box's median chip for that stage (the chips of a CCL
mix are out of phase, so one poll or the power limit is no judge). Limits are only read.
Missing power rows or a burst that never started print what is known; it never fails (exit 0).
"""
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime

from chunk_sched import load as load_rows
from chunk_sched import nchunks, tray_of
from fleet_py import BDF_RE, c8_faults  # fault cut-off: first-fault epochs from fault_times.tsv and in-sample

AT_LIMIT = 0.97  # mean power >= 97% of TDP or mean current >= 97% of TDC
TDP_RULE = 95.0  # chunks 2-5 whose chips average >= 95% of their own TDP are power-limited, not short
MED_WARN = 0.50  # a chip whose median W over the stage's polls is under 50% of the box's median chip is a
# LOAD WARN naming it by BDF and tray 1-4 (never a FAIL); its own power limit is not part of the judgment
MED_MIN_N = (
    10  # polls a chip needs in the stage for that median (~6 s apart): a 4-5 poll quiet-burst's median is one phase
)
THROTTLE = 0.97  # a chip whose mean clock is under 97% of the stage's top chip clock is clock-throttled
CHIP_MIN_N = 3  # C8 samples a chip needs in the stage (~6 s apart) to be judged
SKIP_S = 5  # each stage's first 5 s (the ramp) is left out of the means
TRAY_LOW = 0.80  # a tray whose chip-sum stays under 80% of its chips' own
# power limits for more than half the stage's load is LOAD LOW: the stage never loaded it (a warn, never a pass).
# Set from run.sh's own C8 records (84 stage records of 46 runs):
# healthy trays' medians run 84-110% (clock-throttled chips hold some heavy-burst trays at 84-90%, which
# 0.90 warned on 12 times); never-loaded trays sit at 28-30%; one box-wide weak heavy burst (176 vs 261
# TFLOP/s per chip) read 74-81%. 0.80 warns on 0 of 324 healthy tray-stages, on 3/3 never-loaded trays and
# on 3/4 trays of that weak burst.
CAP_W = (
    10,
    1000,
)  # a plausible chip power limit in W; power1_max 4293967296 uW (c01u14's sentinel) is a limit not read
CHIP_W_MAX = 1000  # a chip power reading over this (0xffffffff uW = 4295 W) is implausible and ignored


def rows(path):
    out = []
    try:
        for ln in open(path):
            f = ln.split()
            if len(f) < 5:
                continue
            try:
                t, w = float(f[0]), [int(v) for v in f[1:5]]
            except ValueError:  # NA / ERR in a tray column: that poll is skipped
                continue
            psu = [int(v) for v in f[10:14]] if len(f) >= 14 and all(v.isdigit() for v in f[10:14]) else None
            out.append((t, w, psu))
    except OSError:
        pass
    return out


def c8_limits(path):
    """{bdf: (TDP W, TDC A)} from C8's limits file; a limit not read is None."""
    out = {}
    try:
        for ln in open(path):
            f = ln.split()
            if len(f) >= 3:
                v = [float(x) if x.replace(".", "", 1).isdigit() else None for x in f[1:3]]
                out[f[0]] = (v[0] / 1e6 if v[0] else None, v[1] / 1e3 if v[1] else None)
    except OSError:
        pass
    return out


def c8_rows(path):
    """[(epoch, bdf, W, A)] from C8's telemetry; a value not read ("-") or a sentinel (over 2000 W or A) is None."""

    def val(x, scale):
        try:
            v = float(x) / scale
        except ValueError:
            return None
        return v if 0 <= v < 2000 else None

    out = []
    try:
        for ln in open(path):
            f = ln.split()
            if len(f) >= 6:
                try:
                    out.append((float(f[0]), f[1], val(f[3], 1e6), val(f[5], 1e3)))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def short(bdf):
    return bdf[5:] if bdf.count(":") == 2 else bdf


def at_limit(c8, lim, start, end):
    """The power-limit record of the stage [start, end]: chips with C8 samples and a limit, those at it, and each one not
    at it (bdf, tray, mean W, TDP, mean A, TDC). None when C8 or its limits were not read."""
    if not c8 or not lim:
        return None
    s0 = start + SKIP_S if end - start > 2 * SKIP_S else start
    per = {}
    for t, b, w, i in c8:
        if s0 <= t <= end:
            p = per.setdefault(b, ([], []))
            if w is not None:
                p[0].append(w)
            if i is not None:
                p[1].append(i)
    mean = lambda v: sum(v) / len(v) if v else None
    chips, n_at, nolim, not_at, pct = 0, 0, [], [], []
    for b in sorted(per):
        tdp, tdc = lim.get(b, (None, None))
        w, i = mean(per[b][0]), mean(per[b][1])
        if tdp and w is not None:
            pct.append((100.0 * w / tdp, b))
        if not (tdp and w is not None) and not (tdc and i is not None):
            nolim.append(short(b))
            continue
        chips += 1
        if (tdp and w is not None and w >= AT_LIMIT * tdp) or (tdc and i is not None and i >= AT_LIMIT * tdc):
            n_at += 1
        else:
            not_at.append(
                dict(
                    bdf=short(b),
                    tray=tray_of(b),
                    power_w=None if w is None else round(w, 1),
                    tdp_w=tdp,
                    current_a=None if i is None else round(i, 1),
                    tdc_a=tdc,
                )
            )
    lo = min(pct) if pct else None
    return dict(
        chips=chips,
        at_limit=n_at,
        not_at_limit=not_at,
        no_limit=nolim,
        start=start,
        end=end,
        tdp_pct_mean=round(sum(p for p, _ in pct) / len(pct), 1) if pct else None,
        tdp_pct_min=dict(bdf=short(lo[1]), tray=tray_of(lo[1]), pct=round(lo[0], 1)) if lo else None,
    )


def topup_plan(c8, lim, start, end, steps, gain=1.0):
    """The pre-ramp's per-chip top-up: every chip not at its power or current limit
    over [start, end] (at_limit(), the LOAD lines' rule) gets ceil(steps * gain * deficit) more heavy ops per burst of
    `steps`, at least 1 and at most `steps`. deficit = (AT_LIMIT * limit - mean) / mean of the nearer of TDP and TDC, a
    share of the chip's mean draw. Limits are only read, never written.
    Returns (at_limit record or None, [{"bdf", "tray", "extra", "power_w", "tdp_w", "current_a", "tdc_a"}])."""
    r = at_limit(c8, lim, start, end)
    if r is None or steps <= 0:
        return r, []
    out = []
    for c in r["not_at_limit"]:
        d = [
            (AT_LIMIT * lv - v) / v for v, lv in ((c["power_w"], c["tdp_w"]), (c["current_a"], c["tdc_a"])) if v and lv
        ]
        if not d:
            continue
        out.append(dict(c, extra=max(1, min(steps, math.ceil(steps * gain * max(min(d), 0.0))))))
    return r, out


def limit_text(r):
    if r is None:
        return "power limit not checked (no C8 telemetry or limits)"
    if not r["chips"]:
        return "power limit not checked (no C8 sample with a limit in the stage)"
    f = lambda v: "?" if v is None else f"{v:.0f}"
    s = f"{r['at_limit']}/{r['chips']} chips at their power or current limit (mean >= {AT_LIMIT:.0%} of TDP or TDC)"
    if r["not_at_limit"]:
        s += "; not at it: " + ", ".join(
            f"{c['bdf']} (tray {c['tray'] or '?'}) {f(c['power_w'])}/{f(c['tdp_w'])} W "
            f"{f(c['current_a'])}/{f(c['tdc_a'])} A"
            for c in r["not_at_limit"]
        )
    if r["no_limit"]:
        s += f"; {len(r['no_limit'])} chip(s) with no limit read: " + " ".join(r["no_limit"])
    return s


def telem_trays(path, start, end):
    """{tray: [chip-sum W per sweep]} from ub0_repro.py's --telem CSV, sweeps with start <= utc <= end; {} if none."""
    sweeps = {}
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    t = datetime.fromisoformat(r["utc"].replace("Z", "+00:00")).timestamp()
                    tray, w = int(r["tray"]), float(r["power_w"])
                except (KeyError, TypeError, ValueError):
                    continue
                if start <= t <= end and 1 <= tray <= 4:
                    sweeps.setdefault(r["utc"], {}).setdefault(tray, []).append(w)
    except OSError:
        return {}
    out = {}
    for u in sorted(sweeps):
        for tray, ws in sweeps[u].items():
            out.setdefault(tray, []).append(sum(ws))
    return out


def swing_text(telem, rs, start, end):
    """'per-tray swing ...' over [start, end]: chip-sum kW min-max per tray from telem, BMC tray W min-max from rs."""
    tt = telem_trays(telem, start, end)
    if tt:
        s = "per-tray swing (chip sum, 1 s) " + ", ".join(
            f"tray {u} {min(v) / 1000:.2f}-{max(v) / 1000:.2f} kW ({(max(v) - min(v)) / 1000:.2f})"
            for u, v in sorted(tt.items())
        )
    else:
        s = "per-tray swing not read (no telem rows)"
    bm = [w for t, w, _ in rs if start <= t <= end]
    if bm:
        s += "; BMC tray 1-4 " + "/".join(f"{min(w[u] for w in bm)}-{max(w[u] for w in bm)}" for u in range(4)) + " W"
        s += " (sensor cap 2550 W)" if any(max(w[u] for w in bm) >= 2550 for u in range(4)) else ""
    return s


def line(power, result, label, c8=None, limits=None, telem=None):
    try:
        res = json.load(open(result))
        b = res.get("burst") or {}
    except (OSError, ValueError, AttributeError):
        res, b = None, {}
    tf = b.get("tflops_per_chip")
    tail = (
        f"{tf:g} TFLOP/s per chip (matmul FLOPs over a time that includes any gaps)"
        if tf is not None
        else "TFLOP/s not reported (no burst)"
    )
    if b.get("op"):  # the op that ran, and why minimal_matmul did not
        tail += f", op {b['op']}" + (f" (fallback: {b['fallback']})" if b.get("fallback") else "")
    if b.get("load"):  # the load and its gap, and any part of it that fell back
        tail += f", load {b['load']}" + (f" ({b['gap']})" if b.get("gap") else "")
        tail += f" (load fallback: {b['load_fallback']})" if b.get("load_fallback") else ""
    t0, sec = b.get("t0"), b.get("seconds")
    if c8 is not None:
        r = at_limit(c8_rows(c8), c8_limits(limits), t0, t0 + sec) if t0 is not None and sec is not None else None
        tail += "; " + (limit_text(r) if t0 is not None and sec is not None else "power limit not checked (no burst)")
        if r is not None and res is not None:
            res["power_limit"] = r
            tmp = f"{result}.tmp{os.getpid()}"
            json.dump(res, open(tmp, "w"))
            os.replace(tmp, result)
    rs = rows(power)
    if telem and t0 is not None and sec is not None:
        tail += "; " + swing_text(telem, rs, t0 - 25, t0 + sec)
    if t0 is None or sec is None or not rs:
        return f"LOAD {label}: box power not read ({len(rs)} polls), {tail}"
    box = all(p for _, _, p in rs)  # the PSU sum is the box's kW (the tray sensors cap at 2550 W each)
    tot = (lambda w, p: sum(p)) if box else (lambda w, p: sum(w))
    idle = [tot(w, p) for t, w, p in rs if t0 - 25 <= t < t0]
    step = [tot(w, p) for t, w, p in rs if t0 <= t <= t0 + 10]
    burst = [w for t, w, p in rs if t0 <= t <= t0 + sec]
    bsum = [tot(w, p) for t, w, p in rs if t0 <= t <= t0 + sec]
    kw = lambda v: f"{v / 1000:.2f}"
    s = f"LOAD {label}: {'box (PSU0-3 sum)' if box else 'trays' + chr(39) + ' sum'} idle {kw(sum(idle) / len(idle)) if idle else '-'} kW, "
    s += f"peak {kw(max(step)) if step else '-'} kW in the first 10 s (the step), "
    if burst:
        s += f"burst mean {kw(sum(bsum) / len(bsum))} kW ({len(burst)} polls), "
        s += "tray 1-4 peak " + "/".join(str(max(w[u] for w in burst)) for u in range(4)) + " W, "
    else:
        s += "no poll inside the burst, "
    return s + tail


def tdp_text(r, k):
    """The power-limit rule: a chunk whose chips
    average >= TDP_RULE of their own TDP is power-limited at default TDP, whatever the box kW; the lowest chip is named but
    not judged (with ~6 s C8 samples of an idle/busy load one chip's mean swings 10-30%), and chunk 1 holds the ramp."""
    if not r or r.get("tdp_pct_mean") is None:
        return ""
    lo = r["tdp_pct_min"]
    s = (
        f"; chips mean {r['tdp_pct_mean']:.1f}% of their own TDP (lowest {lo['bdf']} (tray {lo['tray'] or '?'}) "
        f"{lo['pct']:.0f}%, not judged)"
    )
    if k == 0:
        return s + ", chunk 1 holds the ramp, not judged"
    return s + (
        f", power-limited at default TDP (goal 8: >= {TDP_RULE:.0f}%)"
        if r["tdp_pct_mean"] >= TDP_RULE
        else f", BELOW {TDP_RULE:.0f}%: the chips have headroom, the load is short (goal 8)"
    )


def kw_why(k, kw, rec, launch_per_s=None, min_kw=None):
    """Why sustained chunk k (1-based) ended under 12 kW: chips
    averaging >= TDP_RULE of their own power limit are a power-limit ceiling at default TDP; below it the load stalled
    (launch rate, the chunk's lowest box kW, the lowest chip). rec: at_limit()'s record. Limits are only read."""
    s = f"LOAD chunk {k}: {kw:.2f} kW < 12 kW target: "
    if not rec or rec.get("tdp_pct_mean") is None:
        return s + "cause not read (no chip power or power-limit telemetry in the chunk)"
    if rec["tdp_pct_mean"] >= TDP_RULE:
        return s + f"power-limit ceiling at default TDP (chips mean {rec['tdp_pct_mean']:.1f}% of limit)"
    lo, f = rec["tdp_pct_min"], lambda v, p: "?" if v is None else f"{v:.{p}f}"  # noqa: E731
    return s + (
        f"load stall (launches/s {'?' if launch_per_s is None else f'{launch_per_s:g}'}, min box kW {f(min_kw, 2)}, "
        f"lowest chip {lo['bdf']} tray {lo['tray'] or '?'} at {lo['pct']:.0f}% of limit)"
    )


def chunk_lines(rows_path, c8, limits, label, out=None):
    """One LOAD line per sustained chunk with its power-limit count and the power-limit chip-mean rule (tdp_text); the records
    go to out (JSON list)."""
    rws = load_rows(rows_path) if os.path.exists(rows_path) else []
    n = nchunks(rws) or "?"
    c8r, lim = c8_rows(c8), c8_limits(limits)
    lines, recs = [], []
    for r in rws:
        if r.get("event") == "chunk" and "start" in r and "end" in r:
            rec = at_limit(c8r, lim, r["start"], r["end"])
            lines.append(
                f"LOAD {label} chunk {r['k'] + 1}/{n} ({(r['end'] - r['start']) / 60:.1f} min): "
                + limit_text(rec)
                + tdp_text(rec, r["k"])
            )
            recs.append(dict(chunk=r["k"] + 1, **(rec or {})))
    if out:
        json.dump(recs, open(out, "w"))
    return lines or [f"LOAD {label}: no chunk ran, power limit not checked"]


def c8_chip_rows(path):
    """[(epoch, bdf, W, A, MHz)] from C8's telemetry (c8_rows() plus the aiclk column); a value not read is None."""

    def val(x, scale):
        try:
            v = float(x) / scale
        except ValueError:
            return None
        return v if 0 <= v < 2000 else None

    out = []
    try:
        for ln in open(path):
            f = ln.split()
            if len(f) >= 6:
                try:
                    out.append((float(f[0]), f[1], val(f[3], 1e6), val(f[5], 1e3), val(f[4], 1)))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def stage_windows(src):
    """The load windows [(start, end)] and the TFLOP/s per chip of one stage: chunk_load.py's rows.jsonl (each chunk) or
    ub0_repro.py's result.json (its burst)."""
    if src.endswith(".jsonl"):
        rws = load_rows(src) if os.path.exists(src) else []
        w = [(r["start"], r["end"]) for r in rws if r.get("event") == "chunk" and "start" in r and "end" in r]
        tf = [(r.get("util") or {}).get("tflops_chip") for r in rws if r.get("event") == "chunk"]
        tf = [x for x in tf if x is not None]
        return w, (round(sum(tf) / len(tf), 1) if tf else None)
    try:
        r = json.load(open(src))
        bs = r.get("cycles") or [r.get("burst") or {}]  # ub0_repro.py --cycles: every cycle's burst
    except (OSError, ValueError, AttributeError):
        return [], None
    w = [(b["t0"], b["t0"] + b["seconds"]) for b in bs if b.get("t0") is not None and b.get("seconds") is not None]
    tf = [b["tflops_per_chip"] for b in bs if b.get("tflops_per_chip") is not None]
    return w, (round(sum(tf) / len(tf), 1) if tf else None)


def chip_lines(c8, lim, wins, label, tflops=None):
    """The per-chip LOAD line of one stage and its LOAD WARN lines. c8: c8_chip_rows(); lim: c8_limits();
    wins: the stage's load windows (each one's first SKIP_S s left out)."""
    if not wins:
        return [f"LOAD {label} chips: no load window, chips not checked"]
    if not c8:
        return [f"LOAD {label} chips: no C8 telemetry, chips not checked"]
    cut = [(s + SKIP_S if e - s > 2 * SKIP_S else s, e) for s, e in wins]
    per = {}
    for t, b, w, i, m in c8:
        if any(s <= t <= e for s, e in cut):
            p = per.setdefault(b, ([], [], []))
            for k, v in enumerate((w, i, m)):
                if v is not None:
                    p[k].append(v)
    if not per:
        return [f"LOAD {label} chips: no C8 sample inside the load, chips not checked"]
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    mm = [mean(per[b][2]) for b in per if per[b][2]]
    top = max(mm) if mm else None
    f = lambda v, u: "?" if v is None else f"{v:.0f}{u}"  # noqa: E731
    parts, warns, hits = [], [], {}
    # each chip's median W over its polls in the stage against the box's median chip (the median of those
    # medians); a dip of one poll, or chips out of phase with each other, moves neither
    med = {b: statistics.median(per[b][0]) for b in per if len(per[b][0]) >= MED_MIN_N}
    box = statistics.median(med.values()) if med else None
    for b in sorted(per):
        tdp, tdc = lim.get(b, (None, None))
        w, i, m = (mean(x) for x in per[b])
        hit = [
            h
            for h, ok in (
                ("power", tdp and w is not None and w >= AT_LIMIT * tdp),
                ("current", tdc and i is not None and i >= AT_LIMIT * tdc),
                ("clock", top and m is not None and m < THROTTLE * top),
            )
            if ok
        ] or ["none"]
        hits["+".join(hit)] = hits.get("+".join(hit), 0) + 1
        parts.append(
            f"{short(b)} (tray {tray_of(b) or '?'}) {f(w, '')}/{f(tdp, '')} W {f(i, '')}/{f(tdc, '')} A {f(m, '')} MHz: "
            + "+".join(hit)
        )
        if b in med and box and med[b] < MED_WARN * box:
            warns.append(
                f"LOAD WARN {short(b)} (tray {tray_of(b) or '?'}): stage {label}: median {med[b]:.0f} W over "
                f"{len(per[b][0])} polls, under {MED_WARN:.0%} of the box median chip {box:.0f} W: the load "
                "does not reach this chip"
            )
    tf = (
        f"{tflops:g} TFLOP/s per chip (stage mean, the same ops on every chip)"
        if tflops is not None
        else "TFLOP/s not reported"
    )
    jd = f"box median chip {box:.0f} W" if box else f"per-chip median not judged (under {MED_MIN_N} polls per chip)"
    s = (
        f"LOAD {label} chips: {len(per)} chips, {tf}; {jd}; limit hit: "
        + ", ".join(f"{k} {v}" for k, v in sorted(hits.items()))
        + "; per chip W/power limit, A/current limit, MHz: "
        + "; ".join(parts)
    )
    return [s] + warns


def own_caps(path):
    """{bdf: (limit W or None, source)}: each chip's own power limit from C8's limits file "<bdf> <power1_max uW>
    <curr1_max mA> [<power1_cap uW>]", power1_cap where it reads plausibly, else power1_max (tt-kmd has no power1_cap).
    Only read, never written (TDP item)."""
    out = {}
    try:
        for ln in open(path):
            f = ln.split()
            if not f:
                continue
            got = None
            for i, src in ((3, "power1_cap"), (1, "power1_max")):
                try:
                    w = float(f[i]) / 1e6
                except (IndexError, ValueError):
                    continue
                if CAP_W[0] <= w <= CAP_W[1]:
                    got = (w, src)
                    break
            out[f[0]] = got or (None, "")
    except OSError:
        pass
    return out


def c8_sweeps(path):
    """C8's rows grouped into sweeps (each sweep reads every chip once, in order): [(epoch, {bdf: W | "unread" |
    "implausible"})]. A reading of "-" is unread; one not a number in 0..CHIP_W_MAX W is implausible (ignored)."""
    out, cur = [], None
    try:
        for ln in open(path):
            f = ln.split()
            if len(f) < 4:
                continue
            try:
                t = float(f[0])
            except ValueError:
                continue
            if cur is None or f[1] in cur[1]:
                cur = (t, {})
                out.append(cur)
            try:
                w = float(f[3]) / 1e6
                cur[1][f[1]] = w if 0 <= w <= CHIP_W_MAX else "implausible"
            except ValueError:
                cur[1][f[1]] = "unread" if f[3] == "-" else "implausible"
    except OSError:
        pass
    return out


def fault_cut(tel_path, faults_path):
    """(per-chip first fault epoch, box-wide cut epoch) for the LOAD LOW/UNREAD judgement. A reading taken at or after a chip's first fault never counts.
    The per-chip epochs come from fleet_py.c8_faults
    (fault_times.tsv lines by BDF, else their tray, else every chip; in-sample: a sentinel or out-of-range reading, or a
    chip missing from a C8 poll round). A HANG line (a CCL stall stalls the whole box) and a fault line naming no chip
    cut every tray at its epoch."""
    first = c8_faults(tel_path, faults_path)
    box = None
    try:
        for r in open(faults_path, errors="replace").read().splitlines():
            ts, _, line = r.partition("\t")
            try:
                t = float(ts)
            except ValueError:
                continue
            if "HANG" in line or not BDF_RE.search(line):
                box = t if box is None else min(box, t)
    except OSError:
        pass
    return first, box


def target_lines(sweeps, caps, wins, label, faults=None, box_cut=None):
    """Per tray, each C8 sweep inside the stage's load (each window's first SKIP_S s left out)
    sums its chips' power and compares it with the sum of their own power limits. Prints one LOAD TARGET line, then per tray:
    LOAD LOW WARN when the sum is under TRAY_LOW in more than half the sweeps; LOAD UNREAD WARN when unread or ignored
    readings leave that open (neither a pass nor a fail). Chip telemetry only: the BMC's Power_UBBn pegs at 2550 W.
    A sweep at or after the first fault of any chip of a tray (a box-wide HANG or bare fault line: every tray)
    never counts, and a tray left with under CHIP_MIN_N pre-fault sweeps reads 'too short to judge', never LOW or UNREAD.
    """
    faults = faults or {}
    if not wins:
        return [f"LOAD TARGET {label}: no load window, trays not checked"]
    cut = [(s + SKIP_S if e - s > 2 * SKIP_S else s, e) for s, e in wins]
    swt = [(t, r) for t, r in sweeps if any(s <= t <= e for s, e in cut)]
    sw = [r for t, r in swt]
    trays = {}
    for b in set(caps) | {b for r in sw for b in r}:
        if tray_of(b):
            trays.setdefault(tray_of(b), set()).add(b)
    if not sw or not trays:
        return [f"LOAD TARGET {label}: no C8 sweep inside the load, trays not checked"]
    if len(sw) < CHIP_MIN_N:
        return [
            f"LOAD TARGET {label}: {len(sw)} C8 sweep(s) inside the load, under {CHIP_MIN_N}: too short to judge, trays not checked"
        ]
    nm = lambda b: f"{short(b)} (tray {u})"  # noqa: E731
    summ, out = [], []
    for u in sorted(trays):
        # a sweep at or after the first fault of any chip of this tray (box-wide cut: every tray) is dropped
        tcut = min(
            (
                t
                for t in [faults.get(b) for b in trays[u]] + ([box_cut] if box_cut is not None else [])
                if t is not None
            ),
            default=None,
        )
        sw_u = [r for t, r in swt if tcut is None or t < tcut]
        if len(sw_u) < CHIP_MIN_N:
            summ.append(
                f"tray {u} too short to judge ({len(sw_u)} pre-fault sweep(s){' of ' + str(len(sw)) + ', cut at its first fault' if tcut is not None else ''})"
            )
            continue
        chips = sorted(trays[u])
        nocap = [b for b in chips if caps.get(b, (None,))[0] is None]
        tgt = sum(caps[b][0] for b in chips) if not nocap else None
        low = ok = 0
        unread, impl, sums, per = {}, {}, [], {b: [] for b in chips}
        for r in sw_u:
            vals = []
            for b in chips:
                v = r.get(b, "unread")
                if v == "unread":
                    unread[b] = unread.get(b, 0) + 1
                elif v == "implausible":
                    impl[b] = impl.get(b, 0) + 1
                else:
                    vals.append(v)
                    per[b].append(v)
            if len(vals) == len(chips) and tgt:
                sums.append(sum(vals))
                low += sum(vals) < TRAY_LOW * tgt
                ok += sum(vals) >= TRAY_LOW * tgt
        n = len(sw_u)
        med = sorted(sums)[len(sums) // 2] if sums else None
        pct = f", median {med:.0f}/{tgt:.0f} W ({med / tgt:.0%})" if med is not None else ""
        cw = "; ".join(
            f"{nm(b)} {sum(per[b]) / len(per[b]):.0f}/{caps[b][0]:.0f} W"
            if per[b] and caps.get(b, (None,))[0]
            else f"{nm(b)} unread"
            for b in chips
        )
        src = "/".join(sorted({caps[b][1] for b in chips if caps.get(b, (None,))[0]})) or "power limit"
        ign = (
            ("; implausible readings ignored: " + ", ".join(f"{nm(b)} {k}x" for b, k in sorted(impl.items())))
            if impl
            else ""
        )
        if low * 2 > n:
            v = "LOW"
            out.append(
                f"LOAD LOW WARN tray {u} ({label}): its {len(chips)} chips summed under {TRAY_LOW:.0%} of their own "
                f"{src} sum ({tgt:.0f} W) in {low} of {n} C8 sweeps of the load{pct}: this stage did not load tray {u} "
                f"to its limit, so its result there is not a clean pass (a warn, not a failed check); per chip mean W/own "
                f"limit: {cw}{ign}"
            )
        elif (n - ok) * 2 > n:  # the unread or ignored sweeps could make it LOW
            v = "UNREAD"
            what = [f"power limit not read for {', '.join(nm(b) for b in nocap)}"] if nocap else []
            what += (
                ["power not read for " + ", ".join(f"{nm(b)} in {k} of {n} sweeps" for b, k in sorted(unread.items()))]
                if unread
                else []
            )
            what += (
                [
                    "implausible power ignored for "
                    + ", ".join(f"{nm(b)} in {k} of {n} sweeps" for b, k in sorted(impl.items()))
                ]
                if impl
                else []
            )
            out.append(
                f"LOAD UNREAD WARN tray {u} ({label}): load at target not judged, only {ok + low} of {n} C8 sweeps "
                f"read in full ({'; '.join(what)}): neither a pass nor a failed check"
            )
        else:
            v = "at target"
        summ.append(f"tray {u} {v} ({src}), under {TRAY_LOW:.0%} in {low}/{n} sweeps{pct}")
    return [
        f"LOAD TARGET {label}: each tray's chip-sum against its chips' own power limits (read only; chip telemetry, "
        f"not the BMC board power): " + "; ".join(summ)
    ] + out


def warn_kn(runs):
    """--mode=long's per-chip LOAD WARN k/N: runs is one load.txt per run; a chip counts once per run that
    has a LOAD WARN naming it, in any stage. A LOAD WARN only: never a FAIL, never numeric or bus-bar."""
    import re

    rx = re.compile(r"^LOAD WARN (\S+) \(tray (\S+)\): stage (.*?): median")
    hit = {}
    for k, path in enumerate(runs):
        try:
            lines = open(path, errors="replace").read().splitlines()
        except OSError:
            lines = []
        for ln in lines:
            m = rx.match(ln)
            if m:
                h = hit.setdefault((m.group(1), m.group(2)), {})
                h.setdefault(k, []).append(m.group(3))
    n = len(runs)
    if not hit:
        return [f"LOAD WARN k/N: no chip under {MED_WARN:.0%} of the box median chip in {n} run(s)"]
    return [
        f"LOAD WARN k/N {len(r)}/{n} {b} (tray {t}): "
        + "; ".join(f"run {k + 1}: " + ", ".join(r[k]) for k in sorted(r))
        for (b, t), r in sorted(hit.items(), key=lambda x: (-len(x[1]), x[0]))
    ]


if __name__ == "__main__":
    if sys.argv[1:2] == ["target"]:
        # argv[6] is $FL_OUT/fault_times.tsv; sweeps at/after a chip's first fault never feed the verdict
        first, box = fault_cut(sys.argv[3], sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else ({}, None)
        print(
            "\n".join(
                target_lines(
                    c8_sweeps(sys.argv[3]),
                    own_caps(sys.argv[4]),
                    stage_windows(sys.argv[2])[0],
                    sys.argv[5],
                    first,
                    box,
                )
            )
        )
    elif sys.argv[1:2] == ["chips"]:
        w, tf = stage_windows(sys.argv[2])
        print("\n".join(chip_lines(c8_chip_rows(sys.argv[3]), c8_limits(sys.argv[4]), w, sys.argv[5], tf)))
    elif sys.argv[1:2] == ["warnkn"]:  # <load.txt of each run...>: per-chip LOAD WARN k/N for --mode=long's summary
        print("\n".join(warn_kn(sys.argv[2:])))
    elif sys.argv[1:2] == ["chunks"]:
        print("\n".join(chunk_lines(*sys.argv[2:7])))
    else:
        print(line(*sys.argv[1:7]))
