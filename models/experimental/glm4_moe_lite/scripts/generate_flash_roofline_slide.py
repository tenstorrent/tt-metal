#!/usr/bin/env python3
"""Generate a 16:9 GLM-4.7-Flash roofline summary slide."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


OUT = Path("/home/tt-admin/sdawle/glm47_flash_wh_glx/GLM-4.7-Flash_Roofline_Summary_Slide.png")

NAVY = "#10192d"
PURPLE = "#7547f5"
BLUE = "#2468b4"
GREEN = "#14825c"
ORANGE = "#d27600"
DARK = "#17202a"
MID = "#5f6c7b"
LIGHT = "#f4f6fb"
GRID = "#dce2eb"


def rounded_box(ax, xy, width, height, *, facecolor, edgecolor=GRID, linewidth=1.2, radius=0.012):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def draw_tt_mark(ax, x: float, y: float, width: float, height: float) -> None:
    """Draw the three polygon Tenstorrent mark from the public SVG geometry."""
    points = [
        [(266.3, 207.5), (180.3, 257.1), (94.3, 207.5), (94.3, 306.8), (180.3, 356.4), (266.3, 306.8)],
        [(266.3, 108.2), (180.3, 157.8), (94.3, 108.2), (94.3, 207.5), (8.3, 157.8), (8.3, 58.5), (94.3, 8.8)],
        [(266.3, 108.2), (266.3, 207.5), (352.3, 157.8), (352.3, 58.5), (266.3, 8.8)],
    ]
    for polygon in points:
        normalized = [(x + (px / 360.0) * width, y + height - (py / 365.0) * height) for px, py in polygon]
        ax.add_patch(Polygon(normalized, closed=True, color=PURPLE))


def metric_card(ax, x, y, w, h, title, throughput, latency, color, note):
    rounded_box(ax, (x, y), w, h, facecolor="white", edgecolor=color, linewidth=2)
    ax.text(x + 0.018, y + h - 0.034, title, fontsize=13, fontweight="bold", color=color, va="top")
    ax.text(x + 0.018, y + h - 0.092, throughput, fontsize=24, fontweight="bold", color=DARK, va="top")
    ax.text(x + 0.018, y + 0.058, latency, fontsize=12, fontweight="bold", color=MID, va="bottom")
    ax.text(x + 0.018, y + 0.026, note, fontsize=8.5, color=MID, va="bottom")


def main() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Template header.
    ax.add_patch(plt.Rectangle((0, 0.845), 1, 0.155, color=NAVY))
    ax.add_patch(plt.Rectangle((0, 0.845), 0.012, 0.155, color=PURPLE))
    ax.text(0.04, 0.922, "GLM 4.7 Flash WH GLX", color="white", fontsize=21, fontweight="bold", va="center")
    draw_tt_mark(ax, 0.922, 0.862, 0.065, 0.12)

    # Slide heading.
    ax.text(
        0.045, 0.792, "Decode roofline: optimized_v1 vs theoretical limits", fontsize=23, fontweight="bold", color=DARK
    )
    ax.text(
        0.045,
        0.754,
        "Batch 1 · ISL 128 · traced steady decode · BF8 dense and experts · BF16 KV cache",
        fontsize=10.5,
        color=MID,
    )
    rounded_box(ax, (0.765, 0.745), 0.19, 0.055, facecolor="#e9f5ef", edgecolor=GREEN)
    ax.text(
        0.86, 0.772, "74.8 → 51.3 ms  (−31.4%)", fontsize=11, fontweight="bold", color=GREEN, ha="center", va="center"
    )

    # Main theoretical ladder.
    ax.text(0.045, 0.685, "Throughput and latency ladder", fontsize=14, fontweight="bold", color=DARK)
    card_y, card_h, card_w = 0.445, 0.205, 0.205
    metric_card(
        ax,
        0.045,
        card_y,
        card_w,
        card_h,
        "Current optimized_v1",
        "19.5 tok/s",
        "51.3 ms/token",
        BLUE,
        "Measured end-to-end",
    )
    metric_card(
        ax,
        0.285,
        card_y,
        card_w,
        card_h,
        "Practical MoE target",
        "56–67 tok/s",
        "15–18 ms/token",
        PURPLE,
        "53% heuristic · 2.9–3.4× current",
    )
    metric_card(
        ax,
        0.525,
        card_y,
        card_w,
        card_h,
        "Hardware ceiling",
        "106–127 tok/s",
        "7.9–9.4 ms/token",
        NAVY,
        "288 GB/s ÷ active weight bytes",
    )

    ax.annotate(
        "", xy=(0.278, 0.548), xytext=(0.255, 0.548), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": GREEN}
    )
    ax.annotate(
        "", xy=(0.518, 0.548), xytext=(0.495, 0.548), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": GREEN}
    )

    # Right-side next steps.
    rounded_box(ax, (0.765, 0.405), 0.19, 0.285, facecolor=LIGHT, edgecolor=GRID)
    ax.text(0.785, 0.655, "Next steps toward the target", fontsize=13, fontweight="bold", color=DARK, va="top")
    steps = [
        ("1", "Re-profile 51.3 ms winner", "Refresh matmul, layout, CCL and attention shares."),
        ("2", "Fuse KV-update layout", "Remove sharding/layout conversion; gate at ≤49 ms."),
        ("3", "Wire GlobalCB prefetch", "Overlap dense weight reads; stretch target 43–47 ms."),
        ("4", "Reduce memory/layout traffic", "Attention sharding and KV-cache partitioning."),
    ]
    sy = 0.606
    for number, title, detail in steps:
        ax.add_patch(plt.Circle((0.79, sy), 0.014, color=PURPLE))
        ax.text(0.79, sy, number, color="white", fontsize=8.5, fontweight="bold", ha="center", va="center")
        ax.text(0.815, sy + 0.011, title, color=DARK, fontsize=9.2, fontweight="bold", va="center")
        ax.text(0.815, sy - 0.014, detail, color=MID, fontsize=7.2, va="center")
        sy -= 0.058

    # Assumptions and interpretation.
    rounded_box(ax, (0.045, 0.24), 0.91, 0.125, facecolor="white", edgecolor=GRID)
    ax.text(0.065, 0.335, "How to interpret the limit", fontsize=12.5, fontweight="bold", color=DARK, va="top")
    ax.text(
        0.065,
        0.298,
        "Critical ASIC payload: 2.277 GB with one selected expert, or 2.711 GB with a two-expert collision.",
        fontsize=9.3,
        color=DARK,
    )
    ax.text(
        0.065,
        0.266,
        "The 53% line is a weight-only MoE heuristic—not a guaranteed end-to-end target. Routing, CCL, attention, norms, layouts and synchronization remain additive.",
        fontsize=9.1,
        color=MID,
    )

    # Bottom takeaway.
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.18, color="#f5f6fb"))
    ax.text(0.045, 0.135, "Key takeaway", fontsize=13, fontweight="bold", color=PURPLE)
    ax.text(
        0.045,
        0.087,
        "optimized_v1 has captured 23.5 ms of latency. The next credible milestone is 47–49 ms; reaching the 15–18 ms weight target requires broader fused/persistent execution.",
        fontsize=11,
        color=DARK,
    )
    ax.text(
        0.955,
        0.027,
        "Source: GLM-4.7-Flash performance brief and sweep_isl_batch_complete_20260724",
        fontsize=7.5,
        color=MID,
        ha="right",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
