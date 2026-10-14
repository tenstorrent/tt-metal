# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compact figure of a matmul_chunk_sweep: production ns per computed token vs chunk size, attention and MLP panels.

    python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_plot.py <summary.csv> <out.png>

summary.csv is matmul_chunk_sweep_report.py's output. Dashed lines mark where the model's matmul config changes; the
shaded band is C < 1024, which the model cannot use.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID, BAND = "#fcfcfb", "#1a1a19", "#5f5e58", "#e6e5df", "#efeee9"
PANELS = [
    (
        "Attention projections",
        [
            ("sliding_qkv", "sliding QKV", "#2a78d6"),
            ("global_qk", "global QK", "#eb6834"),
            ("sliding_o", "sliding o_proj", "#1baf7a"),
            ("global_o", "global o_proj", "#eda100"),
        ],
        [(2304, "1D→2D"), (4096, "LoFi from here"), (12288, "ttnn default")],
    ),
    (
        "MLP projections",
        [("mlp_gate", "gate (+GELU)", "#e87ba4"), ("mlp_up", "up", "#008300"), ("mlp_down", "down", "#4a3aa7")],
        [(2304, "1D→2D"), (12288, "ttnn default")],
    ),
]
MARKERS = ["o", "s", "^", "D"]
VALID_CHUNKS = [1024, 2048, 4096, 8192, 16384]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary")
    p.add_argument("out")
    args = p.parse_args()
    rows = [r for r in csv.DictReader(open(args.summary)) if r["attn_fidelity"] == "prod" and r.get("median_ns")]

    plt.rcParams.update(
        {"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (title, ops, marks) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axvspan(200, 1024 / 1.08, color=BAND, zorder=0)
        ax.text(215, 76, "C < 1024: not a\nvalid model chunk", color=INK_2, fontsize=8, va="bottom")
        for c in VALID_CHUNKS:
            ax.axvline(c, color=GRID, lw=1, zorder=0)
        for x, label in marks:
            ax.axvline(x, color=INK_2, lw=1, ls=(0, (2, 3)), zorder=1)
            ax.text(x / 1.03, 3000, label, color=INK_2, fontsize=8, rotation=90, va="top", ha="right")
        for (op, name, color), marker in zip(ops, MARKERS):
            pts = sorted((int(r["chunk"]), float(r["ns_per_token"])) for r in rows if r["op"] == op)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=4, label=name, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(200, 42000)
        ax.set_ylim(70, 3200)
        ax.set_xticks([256, 512] + VALID_CHUNKS + [32768])
        ax.set_xticklabels(["256", "512", "1K", "2K", "4K", "8K", "16K", "32K"])
        ax.set_xlabel("chunk size C (tokens per request; C/8 rows per device)")
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.grid(True, axis="y", which="both", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        legend = ax.legend(
            frameon=True, fontsize=9, loc="upper right", labelcolor=INK, facecolor=SURFACE, edgecolor=SURFACE
        )
        legend.set_zorder(5)
    axes[0].set_ylabel("ns per computed token (matmul time / (C/8) rows)")
    fig.suptitle(
        "Gemma4-31B prefill matmuls on one Blackhole device (CP8/TP4 shapes, production configs, median of timed calls)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
