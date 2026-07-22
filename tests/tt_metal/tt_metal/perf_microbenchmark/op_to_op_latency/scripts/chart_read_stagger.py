#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-stagger de-correlation chart (from run_read_stagger.sh CSVs).

Left panel (wide sweep, log-x): aggregate BW and cross-core done-spread vs the induced per-core
read-start stagger. Shows the goldilocks point -- a small deliberate read offset de-correlates the
56 lockstep DRAM readers, collapsing NoC contention (done-spread down ~5x, BW up ~25%); past the
optimum the induced delay itself serializes the last core and BW falls.

Right panel (fine sweep, averaged over reps): zoom on the optimum -- completion skew, done-spread,
and worst-core barrier drain vs stagger, confirming the minimum near stagger~32 (~22us induced
start spread, ~0.4us/core across 56 cores).
"""
import argparse
import csv
from pathlib import Path


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: float(v) for k, v in r.items()})
    return rows


def avg_by_stagger(paths):
    acc = {}
    for p in paths:
        for r in load(p):
            acc.setdefault(r["stagger"], []).append(r)
    out = []
    for s in sorted(acc):
        rs = acc[s]
        m = {k: sum(x[k] for x in rs) / len(rs) for k in rs[0]}
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-csv", type=Path, required=True)
    ap.add_argument("--fine-csvs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    wide = sorted(load(args.wide_csv), key=lambda r: r["stagger"])
    fine = avg_by_stagger(args.fine_csvs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---- left: wide sweep, BW + done-spread vs stagger (log-x, but include 0) ----
    xs = [max(r["stagger"], 0.5) for r in wide]  # map 0 -> 0.5 for log axis
    bw = [r["agg_total_gbps"] for r in wide]
    ds = [r["writer_done_spread_us"] for r in wide]
    axL.set_xscale("log")
    l1 = axL.plot(xs, bw, "o-", color="#1e88e5", label="aggregate BW (GB/s)")
    axL.set_xlabel("induced per-core read stagger (spin units; 0 shown at 0.5)")
    axL.set_ylabel("aggregate BW (GB/s)", color="#1e88e5")
    axL.tick_params(axis="y", labelcolor="#1e88e5")
    axL.set_ylim(0, 220)
    axR2 = axL.twinx()
    l2 = axR2.plot(xs, ds, "s--", color="#e53935", label="done-spread (us)")
    axR2.set_ylabel("cross-core done-spread (us)", color="#e53935")
    axR2.tick_params(axis="y", labelcolor="#e53935")
    axL.set_title("read-stagger de-correlation: full sweep (56c, balanced math)")
    # mark the peak-BW point
    bi = max(range(len(bw)), key=lambda i: bw[i])
    axL.annotate(
        f"peak {bw[bi]:.0f} GB/s\n@ stagger={int(wide[bi]['stagger'])}",
        xy=(xs[bi], bw[bi]),
        xytext=(xs[bi], bw[bi] - 70),
        ha="center",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    axL.legend(l1 + l2, [h.get_label() for h in l1 + l2], loc="lower center", fontsize=8)

    # ---- right: fine sweep, skew metrics vs stagger (linear) ----
    fx = [r["stagger"] for r in fine]
    axR.plot(fx, [r["read_compl_skew_us"] for r in fine], "o-", color="#2e7d32", label="read completion skew")
    axR.plot(fx, [r["writer_done_spread_us"] for r in fine], "s-", color="#e53935", label="writer done-spread")
    axR.plot(fx, [r["write_drain_max_us"] for r in fine], "^-", color="#c62828", label="worst-core barrier drain")
    axR.plot(fx, [r["read_start_skew_us"] for r in fine], "d--", color="#9e9e9e", label="induced start skew")
    axR.set_xlabel("induced per-core read stagger (spin units)")
    axR.set_ylabel("microseconds")
    axR.set_title("zoom on optimum (avg of reps): skew / drain vs stagger")
    axR.grid(alpha=0.25)
    axR.legend(fontsize=8)
    bi2 = min(range(len(fine)), key=lambda i: fine[i]["writer_done_spread_us"])
    axR.axvline(fx[bi2], color="#1e88e5", ls=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print(f"chart -> {args.out_png}")
    print(
        f"wide peak BW {bw[bi]:.1f} GB/s @ stagger {int(wide[bi]['stagger'])}; "
        f"baseline {wide[0]['agg_total_gbps']:.1f} GB/s @ stagger 0"
    )
    print(
        f"fine done-spread min {fine[bi2]['writer_done_spread_us']:.1f}us @ stagger {int(fx[bi2])} "
        f"(baseline {fine[0]['writer_done_spread_us']:.1f}us)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
