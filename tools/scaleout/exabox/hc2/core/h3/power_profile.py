#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""h3/power_profile.py: box power profile from the reliability telemetry logger's 1 s rows (h3/power-profile.sh).

  power_profile.py summarize --tel <telemetry dir> --out <dir> [--marks <marks.tsv>] [--gens <gens.log>]... [--label <s>]
  power_profile.py compare <a/profile.json> <b/profile.json> [...]      markdown table, first file is the reference

Sources (telemetry normal.tsv and its rotated normal.<n>.gz, 60 s means; burst.tsv raw samples where a high-rate window was open): BMC Power_Total (else the sum of Power_PSU*) as box kW,
BMC Power_UBB<n>_ASIC<m> summed over chips as chip kW (sideband; no in-band chip read), HSC hsc_iout summed per tray, and tray W from
tray_pin (READ_PIN of HSC 0x40 + 0x42, the per-tray power of record), else (per timestamp and tray) the hsc_pin feed sum. Power_UBB<n>
clips at 2550 W: each window reports the fraction of its BMC seconds with a UBB at the cap (bmc_sat rows, else a Power_UBB
reading at or above 2550 W in older logs). Load and idle windows
come from marks.tsv ('<epoch>\\t<load start ...|load end ...|idle start|idle end>'), gens from 'H3 GEN <n> <start> <end>
<s> ok|fail:<why>' lines (h3.sh gens.log; any log with that line shape). Without marks the whole log is the load window.
Writes <out>/profile.csv (one row per second), <out>/profile.json and, where matplotlib imports, <out>/profile.png.
"""
import argparse
import collections
import glob
import gzip
import json
import math
import os
import re
import sys


def rows(tel):
    files = sorted(glob.glob(os.path.join(tel, "normal.*.gz")), key=lambda p: int(re.findall(r"\.(\d+)\.gz$", p)[0]))
    files.append(os.path.join(tel, "normal.tsv"))
    for f in files:
        if not os.path.exists(f):
            continue
        with gzip.open(f, "rt") if f.endswith(".gz") else open(f) as fh:
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) != 8 or not p[0].isdigit():
                    continue
                try:
                    yield int(p[0]), p[2], p[3], float(p[6])
                except ValueError:
                    continue
    # burst.tsv (and its rotated burst.tsv.<n>.gz): every raw sample while a high-rate window is open, about 1-5 s per
    # sensor under load. Read after the 60 s normal rows, so a burst sample replaces the minute mean in its second.
    files = sorted(
        glob.glob(os.path.join(tel, "burst.tsv.*.gz")), key=lambda p: -int(re.findall(r"\.(\d+)\.gz$", p)[0])
    )
    files.append(os.path.join(tel, "burst.tsv"))
    for f in files:
        if not os.path.exists(f):
            continue
        with gzip.open(f, "rt") if f.endswith(".gz") else open(f) as fh:
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) != 4 or ln.startswith("#"):
                    continue
                try:
                    yield int(float(p[0])), p[1], p[2], float(p[3])
                except ValueError:
                    continue


UBB_CAP_W = 2550


def tray_of_feed(key):  # 't3.0x40' -> 3, tray_pin 't3' -> 3
    m = re.match(r"t(\d)(\.|$)", key)
    return int(m.group(1)) if m else None


def series(tel):
    """box kW, chip kW, tray A, tray W, chips, and ubb {t: (UBBs read, UBBs at the cap)}."""
    box, psu, chip, ia, pw = (
        {},
        collections.defaultdict(dict),
        collections.defaultdict(dict),
        collections.defaultdict(lambda: collections.defaultdict(dict)),
        collections.defaultdict(lambda: collections.defaultdict(dict)),
    )
    tp, ubb, sat = collections.defaultdict(dict), collections.defaultdict(set), collections.defaultdict(set)
    for t, kind, key, v in rows(tel):
        if kind == "bmc" and key == "Power_Total":
            box[t] = v / 1000.0
        elif kind == "bmc" and re.fullmatch(r"Power_PSU\d+", key):
            psu[t][key] = v
        elif kind == "bmc" and re.fullmatch(r"Power_UBB\d+_ASIC\d+", key):
            chip[t][key] = v
        elif kind == "bmc" and re.fullmatch(r"Power_UBB\d+", key):
            ubb[t].add(key)
            if v >= UBB_CAP_W:
                sat[t].add(key)
        elif kind == "bmc_sat" and v:
            sat[t].add(key)
        elif kind == "tray_pin":
            tr = tray_of_feed(key)
            if tr:
                tp[t][tr] = v
        elif kind in ("hsc_iout", "hsc_pin"):
            tr = tray_of_feed(key)
            if tr:
                (ia if kind == "hsc_iout" else pw)[t][tr][key] = v
    if not box and psu:
        box = {t: sum(d.values()) / 1000.0 for t, d in psu.items()}
    nchip = max((len(d) for d in chip.values()), default=0)
    chipk = {t: sum(d.values()) / 1000.0 for t, d in chip.items() if nchip and len(d) >= nchip - 1}
    tray_a = {t: {tr: sum(f.values()) for tr, f in d.items()} for t, d in ia.items()}
    # per (t, tray): tray_pin (telemetry from 31d28f89 on) where that t has it, else the feed sum (older logs, mixed files)
    tray_w = {t: {tr: sum(f.values()) for tr, f in d.items()} for t, d in pw.items()}
    for t, d in tp.items():
        tray_w.setdefault(t, {}).update(d)
    ubbs = {t: (ubb[t] | sat[t], sat[t]) for t in set(ubb) | set(sat)}
    return box, chipk, tray_a, tray_w, nchip, ubbs


def ffill(d, ts, max_gap=15):
    """value at each t in ts: the last sample at or before t, if no older than max_gap s."""
    keys = sorted(d)
    out, i, last = {}, 0, None
    for t in ts:
        while i < len(keys) and keys[i] <= t:
            last = keys[i]
            i += 1
        if last is not None and t - last <= max_gap:
            out[t] = d[last]
    return out


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 3),
        "p95": round(pct(xs, 0.95), 3),
        "peak": round(max(xs), 3),
        "min": round(min(xs), 3),
    }


def windows(marks, t0, t1):
    load, idle, op = [], [], {}
    if marks and os.path.exists(marks):
        for ln in open(marks):
            p = ln.rstrip("\n").split("\t", 1)
            if len(p) != 2 or not p[0].isdigit():
                continue
            t, what = int(p[0]), p[1]
            for kind, lst in (("load", load), ("idle", idle)):
                if what.startswith(kind + " start"):
                    op[kind] = (t, what[len(kind) + 7 :].strip())
                elif what.startswith(kind + " end") and kind in op:
                    lst.append((op[kind][0], t, op.pop(kind)[1]))
    if not load:
        load = [(t0, t1, "whole log")]
    return load, idle


GEN_RE = re.compile(r"H3 GEN (\d+) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) ([\d.]+) (\S+)")
CASE_RE = re.compile(r"H3 CASE (\d+) (\S+)")


def gens(paths):
    out = []
    for p in paths:
        case = {}
        for ln in open(p, errors="replace"):
            m = CASE_RE.search(ln)
            if m:
                case[m.group(1)] = m.group(2)
            m = GEN_RE.search(ln)
            if m:
                out.append(
                    {
                        "n": int(m.group(1)),
                        "start": float(m.group(2)),
                        "end": float(m.group(3)),
                        "s": float(m.group(4)),
                        "ok": m.group(5) == "ok",
                        "case": case.get(m.group(1), ""),
                        "log": p,
                    }
                )
    return out


def summarize(a):
    box, chipk, tray_a, tray_w, nchip, ubbs = series(a.tel)
    allt = sorted(set(box) | set(chipk) | set(tray_a))
    if not allt:
        print(f"power_profile: no telemetry rows in {a.tel}", file=sys.stderr)
        return 1
    t0, t1 = allt[0], allt[-1]
    ts = list(range(t0, t1 + 1))
    B, C, A, W = ffill(box, ts), ffill(chipk, ts, 5), ffill(tray_a, ts, 5), ffill(tray_w, ts, 5)
    U = ffill(ubbs, ts, 15)
    trays = sorted({tr for d in list(tray_a.values()) + list(tray_w.values()) for tr in d})
    load, idle = windows(a.marks, t0, t1)
    G = gens(a.gens or [])
    in_gen = set()
    for g in G:
        in_gen.update(range(int(g["start"]), int(g["end"]) + 1))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "profile.csv"), "w") as f:
        f.write("epoch,phase,in_gen,box_kw,chip_kw," + ",".join(f"tray{tr}_a,tray{tr}_w" for tr in trays) + "\n")
        for t in ts:
            ph = next((w[2] or "load" for w in load if w[0] <= t <= w[1]), None) or (
                "idle" if any(w[0] <= t <= w[1] for w in idle) else ""
            )
            cells = [
                str(t),
                ph.replace(",", ";"),
                "1" if t in in_gen else "0",
                f"{B[t]:.3f}" if t in B else "",
                f"{C[t]:.3f}" if t in C else "",
            ]
            for tr in trays:
                cells += [
                    f"{A[t][tr]:.1f}" if t in A and tr in A[t] else "",
                    f"{W[t][tr]:.0f}" if t in W and tr in W[t] else "",
                ]
            f.write(",".join(cells) + "\n")

    def win(sel):
        tt = [t for t in ts if sel(t)]
        r = {"seconds": len(tt), "box_kw": stats([B.get(t) for t in tt]), "chip_kw": stats([C.get(t) for t in tt])}
        r["tray_a"] = {str(tr): stats([A[t].get(tr) for t in tt if t in A]) for tr in trays}
        r["tray_w"] = {str(tr): stats([W[t].get(tr) for t in tt if t in W]) for tr in trays}
        r["tray_w_sum"] = stats([sum(W[t].values()) for t in tt if t in W and len(W[t]) == len(trays)])
        ut = [t for t in tt if t in U]
        r["ubb_sat"] = {
            "seconds": len(ut),
            "frac": round(sum(1 for t in ut if U[t][1]) / len(ut), 3) if ut else None,
            "by_ubb": {
                k: round(sum(1 for t in ut if k in U[t][1]) / len(ut), 3)
                for k in sorted({k for t in ut for k in U[t][0]})
            },
        }
        return r

    in_load = lambda t: any(w[0] <= t <= w[1] for w in load)
    prof = {
        "label": a.label or os.path.basename(os.path.abspath(a.out)),
        "tel": os.path.abspath(a.tel),
        "box_source": "bmc Power_Total" if any(True for _ in box) else "none",
        "chips": nchip,
        "window": [t0, t1],
        "load_windows": [list(w) for w in load],
        "idle_windows": [list(w) for w in idle],
        "load": win(in_load),
        "idle": win(lambda t: any(w[0] <= t <= w[1] for w in idle)),
        "in_gen": win(lambda t: t in in_gen and in_load(t)),
        "between_gens": win(lambda t: in_load(t) and t not in in_gen),
    }
    lsec = sum(w[1] - w[0] for w in load) or 1
    ok = [g for g in G if g["ok"]]
    bycase = collections.defaultdict(list)
    for g in ok:
        bycase[g["case"] or "?"].append(g["s"])
    prof["gens"] = {
        "n": len(G),
        "ok": len(ok),
        "failed": len(G) - len(ok),
        "mean_s": round(sum(g["s"] for g in ok) / len(ok), 1) if ok else None,
        "by_case": {k: {"n": len(v), "mean_s": round(sum(v) / len(v), 1)} for k, v in sorted(bycase.items())},
        "duty_cycle": round(sum(g["s"] for g in G) / lsec, 3) if G else None,
    }
    with open(os.path.join(a.out, "profile.json"), "w") as f:
        json.dump(prof, f, indent=1)
    L, I = prof["load"], prof["idle"]

    def g(d, k):
        return d.get(k, "-") if d.get("n") else "-"

    print(
        f"profile {prof['label']}: load {L['seconds']} s, box kW mean {g(L['box_kw'],'mean')} p95 {g(L['box_kw'],'p95')} "
        f"peak {g(L['box_kw'],'peak')}, idle {g(I['box_kw'],'mean')}; chip kW mean {g(L['chip_kw'],'mean')} "
        f"p95 {g(L['chip_kw'],'p95')}; gens {prof['gens']['ok']}/{prof['gens']['n']} mean {prof['gens']['mean_s']} s, "
        f"duty {prof['gens']['duty_cycle']}; tray W sum mean {g(L['tray_w_sum'],'mean')}, "
        f"UBB at 2550 W cap {L['ubb_sat']['frac'] if L['ubb_sat']['frac'] is not None else '-'} of load s"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot([t - t0 for t in ts if t in B], [B[t] for t in ts if t in B], label="box kW (BMC)", lw=0.8)
        ax.plot([t - t0 for t in ts if t in C], [C[t] for t in ts if t in C], label="chip kW (BMC per-ASIC)", lw=0.8)
        for gg in G:
            ax.axvspan(gg["start"] - t0, gg["end"] - t0, color="g" if gg["ok"] else "r", alpha=0.08)
        ax.set_xlabel("s")
        ax.set_ylabel("kW")
        ax.set_title(prof["label"])
        ax.legend()
        fig.savefig(os.path.join(a.out, "profile.png"), dpi=110, bbox_inches="tight")
        print(f"plot {os.path.join(a.out, 'profile.png')}")
    except Exception as e:  # noqa: BLE001 - the plot is optional
        print(f"no plot ({type(e).__name__}: {e})")
    return 0


def compare(paths):
    P = [json.load(open(p)) for p in paths]

    def v(p, *ks):
        d = p
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else None

    rows_ = [
        ("box kW mean (load)", ("load", "box_kw", "mean")),
        ("box kW p95 (load)", ("load", "box_kw", "p95")),
        ("box kW peak (load)", ("load", "box_kw", "peak")),
        ("box kW in gens", ("in_gen", "box_kw", "mean")),
        ("box kW between gens", ("between_gens", "box_kw", "mean")),
        ("idle kW", ("idle", "box_kw", "mean")),
        ("chip kW mean (load)", ("load", "chip_kw", "mean")),
        ("chip kW p95 (load)", ("load", "chip_kw", "p95")),
        ("gen s mean", ("gens", "mean_s")),
        ("duty cycle", ("gens", "duty_cycle")),
    ]
    trays = sorted({tr for p in P for tr in (p.get("load", {}).get("tray_a") or {})})
    rows_ += [(f"tray {tr} A mean (load)", ("load", "tray_a", tr, "mean")) for tr in trays]
    wtrays = sorted({tr for p in P for tr in (p.get("load", {}).get("tray_w") or {})})
    for tr in wtrays:
        rows_ += [
            (f"tray {tr} W mean (load)", ("load", "tray_w", tr, "mean")),
            (f"tray {tr} W p95 (load)", ("load", "tray_w", tr, "p95")),
        ]
    if wtrays:
        rows_ += [
            ("tray W sum mean (load)", ("load", "tray_w_sum", "mean")),
            ("tray W sum p95 (load)", ("load", "tray_w_sum", "p95")),
        ]
    rows_ += [("UBB at 2550 W cap, fraction of load s", ("load", "ubb_sat", "frac"))]
    print("| metric | " + " | ".join(p["label"] for p in P) + (" | Δ vs first |" if len(P) > 1 else " |"))
    print("|---" * (len(P) + 1 + (len(P) > 1)) + "|")
    for name, ks in rows_:
        vals = [v(p, *ks) for p in P]
        d = ""
        if len(P) > 1:
            d = " | " + ", ".join(
                f"{(x - vals[0]) / vals[0] * 100:+.0f}%" if isinstance(x, (int, float)) and vals[0] else "-"
                for x in vals[1:]
            )
        print(f"| {name} | " + " | ".join("-" if x is None else str(x) for x in vals) + d + " |")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("summarize")
    s.add_argument("--tel", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--marks")
    s.add_argument("--gens", action="append")
    s.add_argument("--label")
    c = sp.add_parser("compare")
    c.add_argument("paths", nargs="+")
    a = ap.parse_args()
    return summarize(a) if a.cmd == "summarize" else compare(a.paths)


if __name__ == "__main__":
    sys.exit(main())
