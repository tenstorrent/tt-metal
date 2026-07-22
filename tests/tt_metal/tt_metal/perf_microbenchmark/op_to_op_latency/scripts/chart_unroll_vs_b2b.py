#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unroll-vs-back-to-back chart (from run_unroll_vs_b2b.sh CSVs).

Left: total wall-clock for K executions run back-to-back (K programs, barrier + op-to-op sync
between each) vs unrolled into one program (workload repeated K times, one barrier at the very end).
Right: the gain (b2b - unroll) vs number of removed op-to-op boundaries (K-1), with the small-K
linear fit (~us saved per boundary) and the large-K saturation.
"""
import argparse
import csv
from pathlib import Path


def load(paths):
    rows = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            k = int(r["K"])
            rows.setdefault(k, []).append({kk: float(vv) for kk, vv in r.items()})
    out = []
    for k in sorted(rows):
        rs = rows[k]
        m = {kk: sum(x[kk] for x in rs) / len(rs) for kk in rs[0]}
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()
    data = load(args.csvs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = [d["K"] for d in data]
    b2b = [d["b2b_us"] for d in data]
    unr = [d["unroll_us"] for d in data]
    bnd = [d["K"] - 1 for d in data]
    gain = [d["b2b_us"] - d["unroll_us"] for d in data]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

    axL.plot(K, b2b, "o-", color="#e53935", label="back-to-back (K programs, barrier between)")
    axL.plot(K, unr, "s-", color="#1e88e5", label="unrolled (1 program, K reps, no mid-barrier)")
    axL.set_xlabel("K (workload executions)")
    axL.set_ylabel("total trace-replay wall-clock (us)")
    axL.set_title("56c balanced: fusing K ops removes the op-to-op barriers")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.25)

    axR.plot(bnd, gain, "D-", color="#8e24aa", label="measured gain (b2b - unroll)")
    # linear fit through small-K (boundaries <= 7) forced through origin
    sm = [(b, g) for b, g in zip(bnd, gain) if 0 < b <= 7]
    if sm:
        slope = sum(b * g for b, g in sm) / sum(b * b for b, g in sm)
        xs = [0, max(bnd)]
        axR.plot(xs, [slope * x for x in xs], "--", color="#9e9e9e", label=f"small-K fit: ~{slope:.0f} us / boundary")
    axR.set_xlabel("op-to-op boundaries removed (K - 1)")
    axR.set_ylabel("wall-clock saved (us)")
    axR.set_title("saving per removed boundary (+ large-K saturation)")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print(f"chart -> {args.out_png}")
    for d in data:
        pb = (d["b2b_us"] - d["unroll_us"]) / (d["K"] - 1) if d["K"] > 1 else 0
        print(
            f"K={int(d['K']):2d}: b2b={d['b2b_us']:.0f} unroll={d['unroll_us']:.0f} "
            f"gain={d['b2b_us']-d['unroll_us']:.0f}us  per-boundary={pb:.1f}us"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
