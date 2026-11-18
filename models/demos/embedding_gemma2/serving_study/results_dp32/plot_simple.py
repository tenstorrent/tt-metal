"""Simple throughput charts for embeddinggemma-2 DP=32 at 128 tokens per text: one line, x = batch size (requests in
flight across the 32 chips), y = requests/s or tokens/s. Data: ./data (sweep_sat.sh, 16 server processes, 128 tokens).
Usage: python3 plot_simple.py -> simple_req_s.png/.svg, simple_tok_s.png/.svg
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
BATCHES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
SURFACE, TEXT, TEXT2, GRID, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

rows = [json.load(open(HERE / "data" / f"load_k16_t128_c{b}.json"))["rows"][0] for b in BATCHES]


def human(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    if v >= 1e5:
        return f"{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k".replace(".0k", "k")
    return f"{v:.0f}"


def chart(key, ylabel, title, out):
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = [r[key] for r in rows]
    ax.plot(BATCHES, y, color=LINE, lw=2.5, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
    i = max(range(len(y)), key=y.__getitem__)
    ax.annotate(
        f"peak {human(y[i])}",
        (BATCHES[i], y[i]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=TEXT2,
        fontsize=10.5,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_ylim(0, max(y) * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(human))
    ax.set_xlabel("Batch size (requests in flight across 32 chips)", color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.set_title(title, loc="left", color=TEXT, fontsize=14, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    fig.text(
        0.01,
        0.01,
        "google/embeddinggemma-2 on a Blackhole Galaxy (DP=32), 128 tokens per text",
        color=TEXT2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{out}.{ext}", facecolor=SURFACE)
    plt.close(fig)


chart("req_s", "Requests per second", "embeddinggemma-2 DP=32: requests/s vs. batch size", "simple_req_s")
chart("tokens_s", "Tokens per second", "embeddinggemma-2 DP=32: tokens/s vs. batch size", "simple_tok_s")
print("ok")
