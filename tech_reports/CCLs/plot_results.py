#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Plot per-link bandwidth from the CCL benchmark CSVs.

    python tech_reports/CCLs/plot_results.py

Reads every data/results_*.csv, keeps the DRAM rows, and writes one figure per
device count to images/, with a panel per resolved topology.
"""

import glob
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
IMAGE_DIR = HERE / "images"

# Okabe-Ito, chosen for hue separation under colour-vision deficiency.
PALETTE = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"), ("#CC79A7", "D")]


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:g}{unit}"
        n /= 1024


def load():
    paths = sorted(glob.glob(str(DATA_DIR / "results_*.csv")))
    if not paths:
        raise SystemExit(f"no results_*.csv in {DATA_DIR}. Run run_bench.sh first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df[(df["memory"] == "dram") & df["linkbw_gbps"].notna()].sort_values("bytes")


def panel(ax, df, title, line_rate, style):
    # Plotting fastest first gives the legend and the gutter the same order.
    ops = sorted(df["op"].unique(), key=lambda op: -df[df["op"] == op]["linkbw_gbps"].max())

    peaks = []
    for op in ops:
        g = df[df["op"] == op]
        color, marker = style[op]
        ax.plot(g["bytes"], g["linkbw_gbps"], color=color, marker=marker,
                markersize=4, linewidth=1.6, label=op.replace("_", "-").capitalize())
        peaks.append((g["linkbw_gbps"].max(), color))

    ax.axhline(line_rate, color="0.6", linewidth=0.9)
    ax.text(0.99, line_rate, f" max {line_rate:g} GB/s", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.45")

    # Peak labels in the right gutter, nudged apart where series land close.
    y_transform = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    placed = math.inf
    for peak, color in peaks:
        y = min(peak, placed - line_rate * 0.05)
        ax.text(1.02, y, f"{peak:.1f} GB/s", transform=y_transform, va="center", fontsize=8, color=color)
        placed = y

    lo, hi = df["bytes"].min(), df["bytes"].max()
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1, 4)])
    ax.xaxis.set_major_formatter(lambda v, _: human(v))
    ax.set_xlim(lo / 1.5, hi * 1.5)
    ax.set_ylim(0, line_rate * 1.05)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Tensor size (bytes)", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)


def main():
    df = load()
    IMAGE_DIR.mkdir(exist_ok=True)
    arch = df["arch"].iloc[0]
    line_rate = float(df["line_rate_gbps"].iloc[0])
    # Assigned once over the whole run, so an op keeps its colour across figures.
    style = {op: PALETTE[i % len(PALETTE)] for i, op in enumerate(sorted(df["op"].unique()))}

    for n, per_n in df.groupby("n"):
        topologies = sorted(per_n["topology_resolved"].unique())
        fig, axes = plt.subplots(1, len(topologies), figsize=(6.5 * len(topologies), 4.6), squeeze=False)
        for ax, topology in zip(axes[0], topologies):
            panel(ax, per_n[per_n["topology_resolved"] == topology], f"{topology.lower()} topology", line_rate, style)
        axes[0][0].set_ylabel("Per-link bandwidth (GB/s)", fontsize=9)

        fig.suptitle(f"CCL performance - {arch.capitalize()}", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.90, f"{n} devices, {per_n['links'].iloc[0]} links/direction, "
                            f"{per_n['dtype'].iloc[0]}, DRAM", fontsize=9, color="0.35", ha="center")
        # wspace leaves room for each panel's gutter labels.
        fig.subplots_adjust(top=0.78, right=0.82, wspace=0.55)

        dest = IMAGE_DIR / f"linkbw_n{n}.png"
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
