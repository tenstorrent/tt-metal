# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Render the two datatype-sweep Pareto charts (top1_perf_pareto.png, top5_perf_pareto.png).

x-axis = full-model accuracy (top-1 or top-5); y-axis = trace-verified teacher-forcing decode
t/s/u (higher = better). Every evaluated config is a point; the non-dominated Pareto frontier is
drawn through the points; the selected config is marked in red; a vertical dotted line sits at the
minimum allowed accuracy for that chart. Palette + accessibility per the dataviz skill (marker
SHAPE is the primary distinction so identity is never colour-alone).

  python make_pareto.py <sweep_results.json> <out_dir>
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz reference palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"  # passing, non-selected
RED = "#d03b3b"  # selected
ORANGE = "#eb6834"  # failing (below threshold) — paired with 'x' marker (secondary encoding)


def _frontier(points):
    """Accuracy-constrained max-performance envelope: for each evaluated accuracy level x, the best
    (highest) decode t/s/u achieved by any config with accuracy >= x. This is the operational Pareto
    frontier ("if I require at least x accuracy, this is the fastest config I measured"); it is
    non-increasing left->right and always connected. When perf is flat across the accuracy range (no
    accuracy<->throughput tradeoff), it correctly renders as a near-horizontal envelope."""
    xs = sorted({round(p["acc"], 6) for p in points})
    front = []
    for x in xs:
        elig = [p["perf"] for p in points if p["acc"] >= x - 1e-9]
        front.append({"acc": x, "perf": max(elig)})
    return front


def make_chart(rows, metric, threshold, out_path):
    import matplotlib.transforms as mtransforms
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    points = [
        {
            "acc": r[metric] * 100.0,
            "perf": r["perf"],
            "id": r["id"],
            "sel": r.get("selected", False),
            "passes": r.get("passes", True),
        }
        for r in rows
    ]
    thr = threshold * 100.0

    # ---- explicit, sane limits (robust to all-points-share-one-accuracy, e.g. top5==100%) ----
    accs = [p["acc"] for p in points]
    perfs = [p["perf"] for p in points]
    ax_lo = min(accs + [thr])
    ax_hi = max(accs + [thr])
    xspan = max(ax_hi - ax_lo, 2.0)
    ax.set_xlim(ax_lo - xspan * 0.18, ax_hi + xspan * 0.22)
    py_lo, py_hi = min(perfs), max(perfs)
    yspan = max(py_hi - py_lo, 1.0)
    ax.set_ylim(py_lo - yspan * 0.22, py_hi + yspan * 0.20)

    # ---- Pareto frontier: accuracy-constrained max-perf envelope ----
    front = _frontier(points)
    if len({round(p["acc"], 6) for p in points}) >= 2:
        ax.plot([p["acc"] for p in front], [p["perf"] for p in front], color=MUTED, lw=2, ls="-", zorder=2)
    else:
        # all configs share one accuracy (e.g. top5==100%): frontier collapses to the max-perf level;
        # draw it as a horizontal envelope across the axis so the frontier is visible + honest.
        ax.axhline(max(perfs), color=MUTED, lw=2, ls="-", zorder=2)

    # ---- threshold vertical dotted line (label via blended transform: x=data, y=axes) ----
    ax.axvline(thr, color=INK2, lw=1.6, ls=":", zorder=1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        thr,
        0.03,
        f" min {metric} = {threshold*100:.0f}%",
        transform=trans,
        color=INK2,
        fontsize=9,
        va="bottom",
        ha="left",
    )

    # ---- marks: shape carries identity, colour reinforces ----
    for p in points:
        if p["sel"]:
            ax.scatter(p["acc"], p["perf"], s=360, marker="*", color=RED, edgecolors=INK, linewidths=0.8, zorder=6)
        elif not p["passes"]:
            ax.scatter(p["acc"], p["perf"], s=95, marker="x", color=ORANGE, linewidths=2.4, zorder=5)
        else:
            ax.scatter(p["acc"], p["perf"], s=95, marker="o", color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=4)

    # ---- de-collided direct labels: group points sharing (acc, perf~) and fan labels vertically ----
    from collections import defaultdict

    buckets = defaultdict(list)
    for p in points:
        buckets[(round(p["acc"], 3), round(p["perf"], 1))].append(p)
    for (ax_, py_), grp in buckets.items():
        grp = sorted(grp, key=lambda p: (not p["sel"], p["id"]))
        for i, p in enumerate(grp):
            ax.annotate(
                p["id"].split("_")[0] + ("★" if p["sel"] else ""),
                (p["acc"], p["perf"]),
                textcoords="offset points",
                xytext=(9, 4 + i * 11 if len(grp) > 1 else 5),
                fontsize=8,
                color=INK if p["sel"] else INK2,
                fontweight="bold" if p["sel"] else "normal",
                zorder=7,
            )

    ax.set_xlabel(f"full-model {metric} accuracy (%)  →  AIME24 chat, 192-prompt + 100 gen", color=INK2, fontsize=10)
    ax.set_ylabel("trace-verified teacher-forcing decode (t/s/u)  →", color=INK2, fontsize=10)
    ax.set_title(
        f"Laguna-XS-2.1 datatype sweep — {metric} vs decode perf (1×4 Blackhole, batch-1)",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=RED,
            markeredgecolor=INK,
            markersize=16,
            label="selected config (C0 baseline)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BLUE,
            markeredgecolor=SURFACE,
            markersize=10,
            label="evaluated, passes gate",
        ),
        Line2D([0], [0], marker="x", color=ORANGE, markersize=10, lw=0, mew=2.4, label="fails accuracy gate"),
        Line2D([0], [0], color=MUTED, lw=2, label="Pareto frontier (max t/s/u at ≥ x acc)"),
        Line2D([0], [0], color=INK2, lw=1.6, ls=":", label=f"min {metric} = {threshold*100:.0f}% threshold"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95, facecolor=SURFACE, edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    results_path, out_dir = sys.argv[1], sys.argv[2]
    data = json.loads(open(results_path).read())
    rows = data["configs"] if isinstance(data, dict) else data
    # perf metric = trace-verified teacher-forcing decode t/s/u
    for r in rows:
        r["perf"] = r["teacher_decode_tsu"]
    make_chart(rows, "top1", 0.90, f"{out_dir}/top1_perf_pareto.png")
    make_chart(rows, "top5", 0.98, f"{out_dir}/top5_perf_pareto.png")


if __name__ == "__main__":
    main()
