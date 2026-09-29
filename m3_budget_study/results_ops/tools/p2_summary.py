#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2 readout from per_op.csv (host only): router / TP collectives at W = 4096 and 8192, and the depth ops
(ag_kv, ag_idx, indexer, sparse) up to h = 548864.

  p2_summary.py [--per-op per_op.csv] [--misc misc_breakdown.csv] [--out p2_summary.md]

Rows are averaged over the sparse layers of each run (3-6) and over the prose and code inputs of a (W, h)
point; packed forwards are listed on their own. worst = worst chip, mean = chip mean (per_op.csv's columns),
eff = roof / mean (Pavlo's statistic), share = worst / layer worst. Growth = (ms at h=548864 - ms at h=0) per
100k history tokens. routing_setup (misc, not Pavlo's router) comes from misc_breakdown.csv when present.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent.parent
TP_OPS = ["router", "norm_ag", "attn_rs"]
DEPTH_OPS = ["ag_kv", "ag_idx", "indexer", "sparse"]
PAVLO = {  # [2,4] MoE/MSA layer, T=5120 at h=51200 (pavlo_reference.md): zone ms, eff
    "router": (0.207, 0.067),
    "norm_ag": (1.211, 0.417),
    "attn_rs": (0.607, 0.416),
    "ag_kv": (0.235, 0.393),
    "ag_idx": (0.091, 1.0),
    "indexer": (0.285, 0.078),
    "sparse": (3.616, 0.039),
}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, p=3):
    return "" if x is None else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-op", default=str(RES / "per_op.csv"))
    ap.add_argument("--misc", default=str(RES / "misc_breakdown.csv"))
    ap.add_argument("--mesh", default="2x4")
    ap.add_argument("--out", default=str(RES / "p2_summary.md"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.per_op)) if r["layer_type"] == "sparse" and r["mesh"] == args.mesh]
    # (W, h label, forward) -> op -> list of (worst, mean, roof, share)
    pts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        fwd = "packed" if r["B"] != "1" else "single"
        h = r["h"] if fwd == "single" else f"{r['h']} (B={r['B']})"
        pts[(int(r["W"]), h, fwd)][r["op"]].append(
            (f(r["worst_ms"]), f(r["mean_ms"]), f(r["roof_ms"]), f(r["share_of_layer_worst"]))
        )

    def agg(key, op):
        v = pts[key].get(op, [])
        if not v:
            return None
        w, m, ro, sh = (mean([x[i] for x in v]) for i in range(4))
        return dict(worst=w, mean=m, roof=ro, eff=(ro / m if ro and m else None), share=sh)

    rs = defaultdict(list)  # routing_setup ms/layer from misc_breakdown.csv
    if Path(args.misc).is_file():
        for r in csv.DictReader(open(args.misc)):
            if r["layer_type"] == "sparse" and r["group"] == "routing_setup":
                fwd = r["forward"]
                h = r["h"] if fwd == "single" else f"{r['h']} (B={r['B']})"
                rs[(int(r["W"]), h, fwd, r["run_id"])].append((f(r["worst_ms_per_layer"]), f(r["mean_ms_per_layer"])))

    def routing_setup(key):
        runs = [v for (W, h, fwd, _), v in rs.items() if (W, h, fwd) == key]
        if not runs:
            return None, None
        return mean([sum(x[0] for x in v) for v in runs]), mean([sum(x[1] for x in v) for v in runs])

    keys = sorted(pts, key=lambda k: (k[0], k[2] == "packed", int(k[1].split()[0].split("+")[0])))
    out = [
        "# P2 summary: router / TP collectives and the depth ops ([2,4] stage, P0-A zone profiles)",
        "",
        "Source: per_op.csv (layers 3-6, sparse; mean over layers and over the prose/code inputs of a point),",
        "misc_breakdown.csv for routing_setup. ms per layer; worst = worst chip, mean = chip mean; eff = roof / mean",
        "(Pavlo's statistic); share = worst / layer worst. Pavlo's [2,4] reference: T=5120 at h=51200.",
        "",
        "## Router and TP collectives",
        "",
        "| W | h | op | worst ms | mean ms | roof ms | eff | share | Pavlo ms / eff |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for k in keys:
        for op in TP_OPS:
            a = agg(k, op)
            if a:
                p = PAVLO.get(op)
                out.append(
                    f"| {k[0]} | {k[1]} | {op} | {fmt(a['worst'])} | {fmt(a['mean'])} | {fmt(a['roof'])} | "
                    f"{fmt(a['eff'] and a['eff'] * 100, 1)}% | {fmt(a['share'] and a['share'] * 100, 1)}% | "
                    f"{p[0]:.3f} / {p[1]:.1%} |"
                )
        w, m = routing_setup(k)
        if w is not None:
            out.append(f"| {k[0]} | {k[1]} | routing_setup (misc) | {fmt(w)} | {fmt(m)} |  |  |  | in misc |")
    out += [
        "",
        "## Depth ops (ag_kv, ag_idx, indexer, sparse) vs history",
        "",
        "| W | h | op | worst ms | mean ms | roof ms | eff | share | Pavlo ms / eff |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for k in keys:
        for op in DEPTH_OPS:
            a = agg(k, op)
            if a:
                p = PAVLO.get(op)
                out.append(
                    f"| {k[0]} | {k[1]} | {op} | {fmt(a['worst'])} | {fmt(a['mean'])} | {fmt(a['roof'])} | "
                    f"{fmt(a['eff'] and a['eff'] * 100, 1)}% | {fmt(a['share'] and a['share'] * 100, 1)}% | "
                    f"{p[0]:.3f} / {p[1]:.1%} |"
                )
    out += ["", "## Growth with history (single forwards, h = 0 -> 548864)", ""]
    out += ["| W | op | ms at h=0 | ms at 139264 | ms at 548864 | mean ms per 100k tokens | eff at 548864 |"]
    out += ["|---:|---|---:|---:|---:|---:|---:|"]
    for W in sorted({k[0] for k in keys}):
        for op in DEPTH_OPS + TP_OPS + ["layer_total"]:
            a0, a1, a2 = (agg((W, h, "single"), op) for h in ("0", "139264", "548864"))
            if not (a0 and a2):
                continue
            slope = (a2["mean"] - a0["mean"]) / 5.48864
            out.append(
                f"| {W} | {op} | {fmt(a0['mean'])} | {fmt(a1 and a1['mean'])} | {fmt(a2['mean'])} | {slope:+.3f} | "
                f"{fmt(a2['eff'] and a2['eff'] * 100, 1)}% |"
            )
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"[p2] {len(keys)} points -> {args.out}")


if __name__ == "__main__":
    main()
