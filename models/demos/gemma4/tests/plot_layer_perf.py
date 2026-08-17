# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plot how a gemma4 decoder layer's cost scales with chunk depth.

Reads a ``sweep_layer_perf.py`` run directory and writes two figures:

``global_scaling.png``
    Per-chunk cost of one global and one sliding decoder layer, with a linear fit on
    the global series. This is the headline: the global layer attends the whole prefix
    so its ring gather grows with depth, while the sliding layer stays inside a
    1024-token window. Both on ONE axis, because the gap between them IS the finding —
    a second y-scale would flatter the sliding series into looking comparable.

``global_op_breakdown.png``
    Where the global layer's growth goes, per chunk, from the ``tt-perf-report`` tables.
    Stacked by op, biggest contributors named and the tail folded into "other".

Data comes from the per-chunk cells (``chunkNNN/{global,sliding}.json`` and
``chunkNNN/global.perf.csv``), so this works on a sweep that is still running and picks
up whatever has landed so far. ``timings.csv`` is used instead when present, since it
covers every chunk rather than only the profiled ones.

Usage::

    python -m models.demos.gemma4.tests.plot_layer_perf
    python -m models.demos.gemma4.tests.plot_layer_perf --run generated/gemma4_layer_perf/full_256k
    python -m models.demos.gemma4.tests.plot_layer_perf --out /tmp/plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHUNK_TOKENS = 4096
DEFAULT_RUN = Path("generated/gemma4_layer_perf/full_256k")

# Slots 1, 2 and 3 of the data-viz reference palette, light mode, used unchanged. Series
# colours carry identity only; every piece of text below wears an ink colour instead, so
# identity is never conveyed by colour alone.
SERIES = {"global": "#2a78d6", "sliding": "#eb6834"}
OP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER_COLOR = "#8a8985"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"

# A 9th op group is never a generated hue: the tail folds into "other".
MAX_OP_GROUPS = len(OP_COLORS)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_MUTED,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 13,
            "legend.frameon": False,
        }
    )


def _recede(ax):
    """Grid and spines behind the data, and only where they help read a value."""
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(False)


def _frame(ax, ymin, ymax):
    """Fit the y-axis to the data, with extra room at the top for the legend.

    Padding is proportional to the SPREAD of the series, not to their magnitude, and the
    lower bound is clamped at zero. That makes the axis do the right thing in both cases
    without a per-chart switch: a series running 0.2 -> 15 ms keeps an effectively
    zero-based axis, while a flat series sitting at ~3 ms gets cropped to its own range
    instead of being squashed into a band at the top of an empty plot.

    The top gets more padding than the bottom so the upper-left legend has clear space.
    """
    spread = max(ymax - ymin, 1e-9)
    lo = max(0.0, ymin - 0.15 * spread)
    hi = ymax + 0.45 * spread
    ax.set_ylim(lo, hi)
    ax.margins(x=0.08)
    ax.legend(loc="upper left", labelcolor=INK_MUTED)


def load_curve(run: Path):
    """``{layer_type: {chunk_idx: (ms, noisy)}}`` from timings.csv, else the cell JSONs."""
    curve = {"global": {}, "sliding": {}}
    source = None

    timings = run / "timings.csv"
    if timings.is_file():
        for row in csv.DictReader(open(timings)):
            tag = row.get("layer_type")
            if tag in curve and row.get("measured_ms"):
                curve[tag][int(row["chunk_idx"])] = (float(row["measured_ms"]), int(row.get("noisy") or 0))
        source = "timings.csv (every chunk, unprofiled)"

    if not any(curve.values()):
        for cell in sorted(run.glob("chunk*/*.json")):
            if cell.stem not in curve:
                continue
            data = json.loads(cell.read_text())
            measured = data.get("measured") or {}
            if measured.get("measured_ms") is None:
                continue
            curve[cell.stem][int(data["chunk_idx"])] = (
                float(measured["measured_ms"]),
                int(measured.get("noisy") or 0),
            )
        source = "profiled cells (chunkNNN/*.json)"

    return curve, source


def load_op_table(run: Path, tag: str = "global"):
    """``{chunk_idx: {op_code: device_us}}`` from the per-cell tt-perf-report CSVs."""
    per_chunk = {}
    for path in sorted(run.glob(f"chunk*/{tag}.perf.csv")):
        chunk_idx = int(path.parent.name.replace("chunk", ""))
        try:
            rows = list(csv.DictReader(open(path)))
        except OSError:
            continue
        if not rows:
            continue
        op_col = next((c for c in rows[0] if c.strip().upper() == "OP CODE"), None)
        us_col = next((c for c in rows[0] if c.strip().lower() == "device time"), None)
        if not op_col or not us_col:
            continue
        totals = {}
        for row in rows:
            try:
                us = float(row[us_col])
            except (TypeError, ValueError):
                continue
            # Strip the matmul shape suffix so the same op does not split into many groups.
            op = row[op_col].split(" ")[0].replace("DeviceOperation", "")
            totals[op] = totals.get(op, 0.0) + us
        if totals:
            per_chunk[chunk_idx] = totals
    return per_chunk


def plot_scaling(curve, source, out: Path):
    """Global and sliding ms against chunk index, with a fit on global."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _recede(ax)

    fit_note = None
    ymin, ymax = float("inf"), 0.0
    for tag in ("global", "sliding"):
        points = sorted(curve[tag].items())
        if not points:
            continue
        xs = np.array([c for c, _ in points], dtype=float)
        ys = np.array([v[0] for _, v in points], dtype=float)
        ymin, ymax = min(ymin, float(ys.min())), max(ymax, float(ys.max()))
        ax.plot(xs, ys, "-o", color=SERIES[tag], linewidth=2.0, markersize=5.5, label=f"{tag} layer", zorder=3)
        # Direct label at the right end, so identity survives without the legend.
        ax.annotate(
            f"{tag}  {ys[-1]:.1f} ms",
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9.5,
            fontweight="medium",
        )
        # Mark any cell the test flagged as noisy, so a bad sample is visible not hidden.
        noisy = [(c, v[0]) for c, v in points if v[1]]
        if noisy:
            ax.plot(
                [c for c, _ in noisy],
                [v for _, v in noisy],
                "o",
                markerfacecolor="none",
                markeredgecolor=INK,
                markersize=10,
                markeredgewidth=1.4,
                linestyle="none",
                label="flagged noisy",
                zorder=4,
            )
        if tag == "global" and len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            pred = slope * xs + intercept
            ss_res = float(((ys - pred) ** 2).sum())
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
            grid = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(grid, slope * grid + intercept, "--", color=SERIES[tag], linewidth=1.3, alpha=0.55, zorder=2)
            fit_note = (
                f"global fit: {slope:.3f} ms per chunk  "
                f"({slope * 1e5 / CHUNK_TOKENS:.2f} ms per 100k tokens of context),  "
                f"intercept {intercept:.2f} ms,  R² {r2:.4f}"
            )

    ax.set_xlabel("chunk index   (context behind the chunk = index × 4096 tokens)")
    ax.set_ylabel("measured replay (ms)")
    ax.set_title("One decoder layer, cost against chunk depth", color=INK, pad=14, loc="left")
    _frame(ax, ymin, ymax)

    caption = f"source: {source}" + (f"\n{fit_note}" if fit_note else "")
    fig.text(0.005, -0.02, caption, ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return fit_note


def load_op_series(run: Path, op: str = "RingJointSDPA"):
    """``{layer_type: {chunk_idx: device_ms}}`` for a single op code."""
    series = {}
    for tag in ("global", "sliding"):
        table = load_op_table(run, tag)
        series[tag] = {c: ops[op] / 1000.0 for c, ops in table.items() if op in ops}
    return series


def load_op_excluded_series(run: Path, op: str = "RingJointSDPA"):
    """``{layer_type: {chunk_idx: device_ms}}`` for the region total MINUS one op.

    The complement of ``load_op_series``: everything in the measured replay that is not
    that op. Flat here means the op is the only thing that scales with depth, which is a
    claim worth plotting rather than inferring from two separate slopes.
    """
    series = {}
    for tag in ("global", "sliding"):
        table = load_op_table(run, tag)
        series[tag] = {c: (sum(ops.values()) - ops.get(op, 0.0)) / 1000.0 for c, ops in table.items()}
    return series


def plot_op_series(series, op: str, out: Path, title: str | None = None, ylabel: str | None = None):
    """A per-op (or per-op-complement) device time against chunk depth, both layer types.

    Isolating the ring attention is what separates "the layer got slower" from "this op
    got slower": the two layer types run the same graph shape and differ only in what
    the attention has to read, so plotting the op alone shows the depth cost with the
    layer's fixed overhead removed.
    """
    if not any(series.values()):
        return None
    _style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _recede(ax)

    notes = []
    ymin, ymax = float("inf"), 0.0
    for tag in ("global", "sliding"):
        points = sorted(series[tag].items())
        if not points:
            continue
        xs = np.array([c for c, _ in points], dtype=float)
        ys = np.array([v for _, v in points], dtype=float)
        ymin, ymax = min(ymin, float(ys.min())), max(ymax, float(ys.max()))
        ax.plot(xs, ys, "-o", color=SERIES[tag], linewidth=2.0, markersize=5.5, label=f"{tag} layer", zorder=3)
        ax.annotate(
            f"{tag}  {ys[-1]:.2f} ms",
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9.5,
            fontweight="medium",
        )
        growth = f"{ys[-1] / ys[0]:.2f}x" if ys[0] else "n/a"
        if len(xs) >= 3:
            slope, _intercept = np.polyfit(xs, ys, 1)
            notes.append(f"{tag}: {ys[0]:.2f} -> {ys[-1]:.2f} ms ({growth}), {slope:.3f} ms per chunk")
        else:
            notes.append(f"{tag}: {ys[0]:.2f} -> {ys[-1]:.2f} ms ({growth})")

    ax.set_xlabel("chunk index   (context behind the chunk = index × 4096 tokens)")
    ax.set_ylabel(ylabel or f"{op} device time (ms)")
    ax.set_title(title or f"{op} alone, cost against chunk depth", color=INK, pad=14, loc="left")
    _frame(ax, ymin, ymax)

    fig.text(0.005, -0.02, "\n".join(notes), ha="left", va="top", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return notes


def plot_op_breakdown(per_chunk, out: Path, tag: str = "global"):
    """Stacked device time by op, per chunk, for one layer type."""
    if not per_chunk:
        return None
    _style()
    chunks = sorted(per_chunk)
    # Rank ops by their total across chunks; keep the top few, fold the tail into "other"
    # rather than inventing colours for a long tail.
    totals = {}
    for ops in per_chunk.values():
        for op, us in ops.items():
            totals[op] = totals.get(op, 0.0) + us
    ranked = [op for op, _ in sorted(totals.items(), key=lambda kv: -kv[1])]
    named, tail = ranked[:MAX_OP_GROUPS], ranked[MAX_OP_GROUPS:]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    _recede(ax)
    xs = np.arange(len(chunks), dtype=float)
    bottom = np.zeros(len(chunks))
    for i, op in enumerate(named):
        vals = np.array([per_chunk[c].get(op, 0.0) / 1000.0 for c in chunks])
        # 2px surface-coloured edge separates adjacent stacked segments.
        ax.bar(xs, vals, bottom=bottom, width=0.72, color=OP_COLORS[i], edgecolor=SURFACE, linewidth=2.0, label=op)
        bottom += vals
    if tail:
        vals = np.array([sum(per_chunk[c].get(op, 0.0) for op in tail) / 1000.0 for c in chunks])
        ax.bar(
            xs,
            vals,
            bottom=bottom,
            width=0.72,
            color=OTHER_COLOR,
            edgecolor=SURFACE,
            linewidth=2.0,
            label=f"other ({len(tail)} ops)",
        )
        bottom += vals

    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in chunks])
    ax.set_xlabel("chunk index")
    ax.set_ylabel("device time in the measured replay (ms)")
    ax.set_title(f"Where the {tag} layer's time goes, by chunk", color=INK, pad=14, loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, labelcolor=INK_MUTED)
    fig.text(
        0.005,
        -0.02,
        "source: per-cell tt-perf-report tables (device time, 32 devices merged → per-device latency)",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return bottom


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=str(DEFAULT_RUN), help=f"sweep run directory (default {DEFAULT_RUN})")
    parser.add_argument("--out", default=None, help="output directory (default <run>/plots)")
    parser.add_argument(
        "--op",
        default="RingJointSDPA",
        help="op code to isolate in its own figure (default RingJointSDPA). Matched against "
        "the tt-perf-report OP CODE with the DeviceOperation suffix and any shape stripped.",
    )
    args = parser.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        parser.error(f"no such run directory: {run}")
    out_dir = Path(args.out) if args.out else run / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    curve, source = load_curve(run)
    n_cells = sum(len(v) for v in curve.values())
    if not n_cells:
        parser.error(f"no measured cells found under {run} (looked for timings.csv and chunkNNN/*.json)")

    scaling_path = out_dir / "global_scaling.png"
    fit_note = plot_scaling(curve, source, scaling_path)
    print(f"{scaling_path}  ({n_cells} cells from {source})")
    if fit_note:
        print(f"  {fit_note}")

    ops = load_op_table(run, "global")
    if ops:
        breakdown_path = out_dir / "global_op_breakdown.png"
        plot_op_breakdown(ops, breakdown_path, "global")
        print(f"{breakdown_path}  ({len(ops)} profiled chunks)")

        sdpa_path = out_dir / f"{args.op.lower()}_scaling.png"
        notes = plot_op_series(load_op_series(run, args.op), args.op, sdpa_path)
        if notes:
            print(f"{sdpa_path}")
            for note in notes:
                print(f"  {note}")
        else:
            print(f"  (no {args.op} rows found — skipping that figure)")

        rest_path = out_dir / f"layer_minus_{args.op.lower()}.png"
        rest_notes = plot_op_series(
            load_op_excluded_series(run, args.op),
            args.op,
            rest_path,
            title=f"Everything in the layer except {args.op}",
            ylabel=f"device time excluding {args.op} (ms)",
        )
        if rest_notes:
            print(f"{rest_path}")
            for note in rest_notes:
                print(f"  {note}")
    else:
        print("  (no per-op tables yet — skipping the breakdown and per-op figures)")

    # The numbers behind the headline, so the figure never has to be trusted on its own.
    if curve["global"]:
        pts = sorted(curve["global"].items())
        first, last = pts[0], pts[-1]
        print(
            f"  global: {first[1][0]:.2f} ms @chunk{first[0]} -> {last[1][0]:.2f} ms @chunk{last[0]} "
            f"({last[1][0] / first[1][0]:.2f}x)"
        )
    if curve["sliding"]:
        pts = sorted(curve["sliding"].items())
        first, last = pts[0], pts[-1]
        print(
            f"  sliding: {first[1][0]:.2f} ms @chunk{first[0]} -> {last[1][0]:.2f} ms @chunk{last[0]} "
            f"({last[1][0] / first[1][0]:.2f}x)"
        )


if __name__ == "__main__":
    main()
