#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate read BW and reader done-spread vs core count, one line per DRAM transaction size
(2K/4K/8K = --page-size-tiles 1/2/4). Twin-axis: left = BW (solid), right = done-spread (dashed).

Shows (a) BW saturates by ~16 cores and stays flat at every txn size -- bigger txns give NO BW gain
(slightly less), and (b) done-spread grows with cores AND markedly worse with bigger txns. Input is
the CSV from the txn-size sweep: txn_kb,cores,tiles_per_core,agg_read_gbps,reader_done_spread_us.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path):
    data = defaultdict(lambda: {"cores": [], "bw": [], "skew": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            # overall_bw_gbps (total bytes / kernel envelope) is the honest metric; fall back to the
            # legacy union-span agg_read_gbps for old CSVs.
            bw_key = "overall_bw_gbps" if "overall_bw_gbps" in row else "agg_read_gbps"
            sk_key = "done_spread_us" if "done_spread_us" in row else "reader_done_spread_us"
            if row[bw_key] in ("FAIL", "nan", ""):
                continue
            txn = row["txn_kb"]
            data[txn]["cores"].append(int(row["cores"]))
            data[txn]["bw"].append(float(row[bw_key]))
            sk = row[sk_key]
            data[txn]["skew"].append(float(sk) if sk not in ("nan", "FAIL", "") else 0.0)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("/tmp/txnsize_bw.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("/tmp/txnsize_bw_vs_cores.png"))
    ap.add_argument("--title", type=str, default="BW & completion skew vs cores, by DRAM transaction size")
    ap.add_argument("--bw-label", type=str, default="overall BW (GB/s)")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv)
    colors = {"2K": "#1e88e5", "4K": "#2e7d32", "8K": "#8e24aa"}

    fig, axL = plt.subplots(figsize=(11, 6))
    axR = axL.twinx()

    order = [t for t in ("2K", "4K", "8K") if t in data]
    for txn in order:
        d = data[txn]
        c = colors.get(txn, "#555")
        axL.plot(d["cores"], d["bw"], "-o", color=c, lw=2.0, ms=5, label=f"{txn} BW")
        axR.plot(
            d["cores"], d["skew"], "--s", color=c, lw=1.4, ms=4, alpha=0.75, mfc="white", label=f"{txn} done-spread"
        )

    axL.set_xlabel("active cores")
    axL.set_ylabel(f"{args.bw_label}  — solid")
    axR.set_ylabel("done-spread (µs)  — dashed")
    axL.set_ylim(0, 230)
    axR.set_ylim(0, max(max(data[t]["skew"]) for t in order) * 1.1)
    axL.grid(alpha=0.25)
    if args.note:
        axL.text(2, 216, args.note, fontsize=8, color="#333")

    # merged legend
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="center left", fontsize=8, ncol=2, framealpha=0.9)

    axL.set_title(args.title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"chart -> {args.out_png}")


if __name__ == "__main__":
    raise SystemExit(main())
