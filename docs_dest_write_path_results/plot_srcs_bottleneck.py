#!/usr/bin/env python3
"""Draws srcs_bottleneck.png: theoretical and measured cycles per FP32 tile for each stage of the Dest write paths.

Usage: plot_srcs_bottleneck.py [<output folder>]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREY = "#7f7f7f"

TILE_BYTES = 4096

# (stage, peak B/clk label, theoretical cycles per tile, measured cycles per tile or None, path colour)
STAGES = [
    ("Unpacker 0\nL1 → Dest", "2 × 64 =\n128 B/clk", 32, 32.25, BLUE),
    ("Packer 0\nDest → L1", "2 × 64 =\n128 B/clk", 32, None, GREY),
    ("Unpacker 2\nL1 → SrcS", "64 B/clk", 64, 83.75, ORANGE),
    ("SFPU copy\nSrcS → Dest", "64 B/clk", 64, 66.25, ORANGE),
]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    for i, (name, peak, theory, measured, color) in enumerate(STAGES):
        ax.bar(i - width / 2, theory, width, color=color, alpha=0.35, edgecolor=color)
        ax.text(i - width / 2, theory + 1.5, f"{theory}", ha="center", fontsize=9)
        ax.text(i - width / 2, theory / 2, peak, ha="center", va="center", fontsize=7.5)
        if measured is not None:
            ax.bar(i + width / 2, measured, width, color=color)
            ax.text(i + width / 2, measured + 1.5, f"{measured:g}", ha="center", fontsize=9, fontweight="bold")
            ax.text(
                i + width / 2,
                measured / 2,
                f"{TILE_BYTES / measured:.0f}\nB/clk",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(i + width / 2, 3, "not\nmeasured\nalone", ha="center", fontsize=7.5, color="0.4")
    ax.set_xticks(range(len(STAGES)), [s[0] for s in STAGES])
    ax.text(
        0.99,
        0.97,
        "Inside bars: peak B/clk (light), measured B/clk (dark)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.3",
    )
    ax.set_ylabel("Cycles per FP32 tile (lower is better)")
    ax.set_ylim(0, 100)
    ax.set_title("Where the time goes: theoretical (light) vs measured (dark)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        "Bottleneck: 12 reads in flight\n64 → about 48 B/clk",
        xy=(2 + width / 2, 83.75),
        xytext=(0.55, 78),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE),
            plt.Rectangle((0, 0), 1, 1, color=GREY),
        ],
        labels=["Unpack to Dest path", "SrcS path", "Both paths"],
        loc="upper left",
        fontsize=9,
    )
    fig.text(
        0.01,
        0.01,
        "Quasar 1x3 emulator, FP32, 4 tiles per Dest bank, steady-state median. SFPU copy measured with no Unpacker 2 traffic.",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "srcs_bottleneck.png", dpi=150)


if __name__ == "__main__":
    main()
