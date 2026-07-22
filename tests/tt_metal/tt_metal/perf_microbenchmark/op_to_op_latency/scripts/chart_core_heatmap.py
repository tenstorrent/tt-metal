#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spatial heatmap of per-core finish time on the physical grid -- shows WHICH cores straggle.

Each Tensix worker is plotted at its (core_x, core_y) and colored by when it finished (write-barrier
done) relative to the first core's go. Reveals the NoC-distance gradient: cores far from their DRAM
channels finish late (hot), near ones finish early (cool). Input: raw device-profiler CSV.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_op_to_op_profiler_csv import load_profiler_csv, parse_chip_freq_mhz  # noqa: E402
from chart_timeline_k2 import per_core_invocations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--title", default="per-core finish time (NoC-distance stragglers)")
    args = ap.parse_args()

    freq = parse_chip_freq_mhz(args.csv.resolve())
    df = load_profiler_csv(args.csv.resolve())
    inv = {k: v for k, v in per_core_invocations(df).items() if v}
    t0 = min(v[-1]["go"] for v in inv.values())
    pts = [(x, y, (v[-1]["wafter"] - t0) / freq) for (s, x, y), v in inv.items()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fs = [p[2] for p in pts]
    fmin, fmax = min(fs), max(fs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        xs, ys, c=fs, cmap="RdYlGn_r", s=900, marker="s", edgecolor="k", linewidth=0.5, vmin=fmin, vmax=fmax
    )
    for x, y, f in pts:
        ax.text(x, y, f"{f:.0f}", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("core_x (NoC column)")
    ax.set_ylabel("core_y (NoC row)")
    ax.invert_yaxis()  # row 0 at top, like the physical grid
    ax.set_aspect("equal")
    ax.set_title(
        f"{args.title}\n{len(pts)} cores | finish {fmin:.0f}–{fmax:.0f}µs (spread {fmax - fmin:.0f}µs)", fontsize=11
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("finish time since go (µs) — red = late straggler")
    ax.set_xticks(sorted(set(xs)))
    ax.set_yticks(sorted(set(ys)))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"chart -> {args.out_png}")
    # column means for the readout
    import statistics as st

    bycol = {}
    for x, y, f in pts:
        bycol.setdefault(x, []).append(f)
    print("finish µs by column:", {x: round(st.mean(bycol[x])) for x in sorted(bycol)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
