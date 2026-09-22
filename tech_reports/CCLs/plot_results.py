#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Plot per-link bandwidth from the CCL benchmark CSVs.

    python tech_reports/CCLs/plot_results.py

Reads every data/results_*.csv and writes to images/: one bandwidth figure per
architecture and device count, with a panel per resolved topology, plus a
memory config figure wherever an op was measured in both L1 and DRAM.
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
    df = df[df["linkbw_gbps"].notna()]
    # Figure names carry arch and device count only, so a second dtype or packet
    # size would silently overwrite rather than produce its own figure.
    for column in ("dtype", "packet"):
        seen = sorted(df[column].unique())
        if len(seen) > 1:
            raise SystemExit(f"{DATA_DIR} holds more than one {column}: {seen}. "
                             f"Move the runs you are not plotting out of the way.")
    # A line and a ring run both resolve to a line below the wrap threshold, so the
    # same cell can be measured twice. Keep one; a repeat is not a second series.
    df = df.drop_duplicates(["arch", "dtype", "op", "n", "bytes", "memory", "topology_resolved"])
    return df.sort_values("bytes")


def panel(ax, groups, title, line_rate):
    """groups: [(label, color, marker, linestyle, frame)], plotted fastest first
    so the legend and the gutter share one order."""
    groups = sorted(groups, key=lambda g: -g[4]["linkbw_gbps"].max())

    for label, color, marker, linestyle, g in groups:
        ax.plot(g["bytes"], g["linkbw_gbps"], color=color, marker=marker, linestyle=linestyle,
                markersize=4, linewidth=1.6, label=label)

    ax.axhline(line_rate, color="0.6", linewidth=0.9)
    ax.text(0.99, line_rate, f" max {line_rate:g} GB/s", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.45")

    # Peak labels in the right gutter, nudged apart where series land close.
    y_transform = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    placed = math.inf
    for _, color, _, _, g in groups:
        peak = g["linkbw_gbps"].max()
        y = min(peak, placed - line_rate * 0.05)
        ax.text(1.02, y, f"{peak:.1f} GB/s", transform=y_transform, va="center", fontsize=8, color=color)
        placed = y

    lo = min(g["bytes"].min() for *_, g in groups)
    hi = max(g["bytes"].max() for *_, g in groups)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1, 4)])
    ax.xaxis.set_major_formatter(lambda v, _: human(v))
    ax.set_xlim(lo / 1.5, hi * 1.5)
    ax.set_ylim(0, line_rate * 1.05)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Tensor size (bytes)", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)


def figure(panels, title, subtitle, dest):
    """panels: [(panel_title, groups)], one subplot each."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.6), squeeze=False)
    for ax, (panel_title, groups, line_rate) in zip(axes[0], panels):
        panel(ax, groups, panel_title, line_rate)
    axes[0][0].set_ylabel("Per-link bandwidth (GB/s)", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.90, subtitle, fontsize=9, color="0.35", ha="center")
    # wspace leaves room for each panel's gutter labels.
    fig.subplots_adjust(top=0.78, right=0.82, wspace=0.55)
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")


def linkbw_figures(df, style):
    # Every column that can differ between CSVs belongs in the key, or a series
    # ends up with two points on the same x.
    for (arch, dtype, n), per_n in df[df["memory"] == "dram"].groupby(["arch", "dtype", "n"]):
        line_rate = float(per_n["line_rate_gbps"].iloc[0])
        panels = []
        for topology in sorted(per_n["topology_resolved"].unique()):
            per_topo = per_n[per_n["topology_resolved"] == topology]
            groups = [(op.replace("_", "-").capitalize(), *style[op], "-", g)
                      for op, g in per_topo.groupby("op")]
            panels.append((f"{topology.lower()} topology", groups, line_rate))
        figure(panels, f"CCL link bandwidth - {arch.capitalize()}",
               f"{n} devices, {per_n['links'].iloc[0]} links/direction, "
               f"{dtype}, DRAM, {per_n['packet'].iloc[0]} B packet",
               IMAGE_DIR / f"linkbw_{arch}_n{n}.png")


def memcfg_figures(df):
    # Every column that can differ between CSVs belongs in the key. Two values of
    # one on a single series would put two points on every x.
    for (arch, dtype, n, topology), per_n in df.groupby(["arch", "dtype", "n", "topology_resolved"]):
        # Only ops measured in both memory configs say anything about the memory limit.
        ops = [op for op, g in per_n.groupby("op") if {"l1", "dram"} <= set(g["memory"])]
        if not ops:
            continue
        line_rate = float(per_n["line_rate_gbps"].iloc[0])
        # Coloured per (op, memory config) pair: the memory config is the comparison
        # here, and with a single op the op colour would make both series identical.
        pairs = [(op, memory) for op in ops for memory in ("dram", "l1")]
        groups = []
        for i, (op, memory) in enumerate(pairs):
            g = per_n[(per_n["op"] == op) & (per_n["memory"] == memory)]
            color, marker = PALETTE[i % len(PALETTE)]
            label = f"{op.replace('_', '-').capitalize()}, {memory.upper()}"
            groups.append((label, color, marker, "-" if memory == "dram" else "--", g))
        figure([(f"{topology.lower()} topology", groups, line_rate)],
               f"CCL mem config - {arch.capitalize()}",
               f"{n} devices, {per_n['links'].iloc[0]} links/direction, "
               f"{dtype}, {per_n['packet'].iloc[0]} B packet",
               IMAGE_DIR / f"memcfg_{arch}_n{n}.png")


def main():
    df = load()
    IMAGE_DIR.mkdir(exist_ok=True)
    # Assigned once over the whole run, so an op keeps its colour across figures.
    style = {op: PALETTE[i % len(PALETTE)] for i, op in enumerate(sorted(df["op"].unique()))}
    linkbw_figures(df, style)
    memcfg_figures(df)


if __name__ == "__main__":
    main()
