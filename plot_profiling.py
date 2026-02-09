#!/usr/bin/env python3
"""
Unified plotting for Galaxy profiling results.

Usage:
    python3 plot_profiling.py coarse <json_dir> --title "Model" -o output.png
    python3 plot_profiling.py fine   <json_dir> --title "Model" -o output.png

Coarse mode: one row per ISL, horizontal bars for module-level timings.
Fine mode:   one row per (ISL × module), horizontal bars for sub-op timings.
Both:        prefill on the left, decode on the right.
"""

import argparse
import json
import os
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── I/O helpers ──────────────────────────────────────────────────────


def _load(path):
    with open(path) as f:
        return json.load(f)


def _seq_label(path):
    m = re.search(r"(\d+k)", os.path.basename(path), re.IGNORECASE)
    return m.group(1) if m else os.path.basename(path).replace(".json", "")


def _sort_key(path):
    m = re.search(r"(\d+)k", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _resolve_jsons(inputs):
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        paths = glob.glob(os.path.join(inputs[0], "*.json"))
    else:
        paths = list(inputs)
    return sorted(paths, key=_sort_key)


# ── Bar chart primitives ─────────────────────────────────────────────


def _barh(ax, names, times, title, cmap="tab20", fontsize=8):
    """Horizontal bar chart with inline labels."""
    total = sum(times)
    if not names:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    # longest bar at top
    names, times = names[::-1], times[::-1]
    colors = getattr(plt.cm, cmap)(range(len(names)))
    bars = ax.barh(names, times, color=colors, edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, times):
        pct = t / total * 100 if total else 0
        label = f"{t:.1f} ms ({pct:.0f}%)"
        inside = t > total * 0.12
        ax.text(
            bar.get_width() - total * 0.01 if inside else bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="right" if inside else "left",
            fontsize=fontsize,
            color="white" if inside else "black",
            fontweight="bold" if inside else "normal",
        )

    ax.set_xlabel("Time (ms)", fontsize=fontsize)
    ax.set_title(f"{title}  ({total:.1f} ms)", fontsize=fontsize + 2, fontweight="bold")
    ax.tick_params(labelsize=fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Coarse plot ──────────────────────────────────────────────────────


def plot_coarse(paths, title, outfile):
    n = len(paths)
    row_h = max(3.5, 14 / max(n, 1))
    fig, axes = plt.subplots(n, 2, figsize=(20, row_h * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{title} — Coarse Profiling", fontsize=16, fontweight="bold", y=0.995)

    for i, fp in enumerate(paths):
        data = _load(fp)
        label = _seq_label(fp)
        for col, phase in enumerate(("prefill", "decode")):
            pd = data.get(phase, {})
            items = sorted(pd.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
            _barh(
                axes[i][col],
                [n for n, _ in items],
                [v["total_ms"] for _, v in items],
                f"{phase.capitalize()}  ISL={label}",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


# ── Fine plot ────────────────────────────────────────────────────────


def _modules_with_fine(data):
    """Modules that have fine sub-breakdowns, ordered by total time."""
    seen = {}
    for phase in ("prefill", "decode"):
        for name, info in data.get(phase, {}).items():
            if info.get("fine"):
                seen[name] = max(seen.get(name, 0), info.get("total_ms", 0))
    return sorted(seen, key=seen.get, reverse=True)


def plot_fine(paths, title, outfile):
    # Collect (label, data, module) triples — one per plot row
    rows = []
    for fp in paths:
        label, data = _seq_label(fp), _load(fp)
        for mod in _modules_with_fine(data):
            rows.append((label, data, mod))
    if not rows:
        print("No fine-grained data found.")
        return

    # Dynamic row heights based on bar count
    heights = []
    for label, data, mod in rows:
        n_bars = max(len(data.get(p, {}).get(mod, {}).get("fine", {})) + 1 for p in ("prefill", "decode"))
        heights.append(max(1.8, n_bars * 0.38))

    fig = plt.figure(figsize=(20, max(10, sum(heights) * 1.1 + 2)))
    gs = gridspec.GridSpec(len(rows), 2, figure=fig, height_ratios=heights, hspace=0.55, wspace=0.35)
    fig.suptitle(f"{title} — Fine-Grained Profiling", fontsize=16, fontweight="bold", y=0.998)

    prev_label = None
    for i, (label, data, mod) in enumerate(rows):
        # ISL separator
        if label != prev_label:
            ax0 = fig.add_subplot(gs[i, 0])
            ax0.annotate(
                f"━━━  ISL = {label}  ━━━",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(-10, 18),
                textcoords="offset points",
                fontsize=13,
                fontweight="bold",
                color="#1a5276",
                ha="left",
                va="bottom",
            )
        else:
            ax0 = fig.add_subplot(gs[i, 0])

        ax1 = fig.add_subplot(gs[i, 1])

        for ax, phase in [(ax0, "prefill"), (ax1, "decode")]:
            info = data.get(phase, {}).get(mod, {})
            fine = info.get("fine", {})
            parent_ms = info.get("total_ms", 0)

            items = sorted(fine.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
            names = [n for n, _ in items]
            times = [v["total_ms"] for _, v in items]

            other = parent_ms - sum(times)
            if other > 0.05:
                names.append("other")
                times.append(other)

            _barh(ax, names, times, f"{phase.capitalize()} / {mod}  ({parent_ms:.1f} ms)", cmap="Set2", fontsize=7)

        prev_label = label

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Plot Galaxy profiling results.")
    ap.add_argument("mode", choices=["coarse", "fine"])
    ap.add_argument("input", nargs="+", help="JSON files or directory")
    ap.add_argument("--title", default="Model")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    paths = _resolve_jsons(args.input)
    if not paths:
        print("No JSON files found.")
        return

    outfile = args.output or f"profiling_{args.mode}.png"

    if args.mode == "coarse":
        plot_coarse(paths, args.title, outfile)
    else:
        plot_fine(paths, args.title, outfile)


if __name__ == "__main__":
    main()
