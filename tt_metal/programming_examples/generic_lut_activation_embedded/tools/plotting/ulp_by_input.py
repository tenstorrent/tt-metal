#!/usr/bin/env python3
"""Plot ULP error versus input domain from raw hardware dump CSVs.

Input CSVs must contain columns:
  input,output

This intentionally accepts explicit dump paths instead of assuming the retired
generic_lut_activation/data/hardware_outputs layout.
"""

import argparse
import gzip
import importlib
import os
from pathlib import Path
import sys

import numpy as np

try:
    from .style import apply_tufte_style
except ImportError:
    from style import apply_tufte_style


def add_tt_poly_fit_to_path():
    search = []
    env = os.environ.get("TT_POLY_FIT_DIR")
    if env:
        search.append(Path(env))
    search += [
        Path.home() / "workspace" / "tt-polynomial-fitter",
        Path("/localdev") / os.environ.get("USER", "") / "tt-polynomial-fitter",
        Path("/proj_sw/user_dev") / os.environ.get("USER", "") / "tt-polynomial-fitter",
    ]
    for root in search:
        if root.exists():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "competition"))
            return root
    raise SystemExit("ulp_by_input: set TT_POLY_FIT_DIR; tt-polynomial-fitter was not found")


def load_dump(path):
    if str(path).endswith(".npz"):
        with np.load(path) as data:
            if "input" not in data or "output" not in data:
                raise SystemExit(f"ulp_by_input: {path} must have input,output arrays")
            x = np.asarray(data["input"], dtype=np.float32)
            y = np.asarray(data["output"], dtype=np.float32)
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite]

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        data = np.genfromtxt(f, delimiter=",", names=True, invalid_raise=False)
    if data.size == 0:
        return np.array([]), np.array([])
    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    if "input" not in names or "output" not in names:
        raise SystemExit(f"ulp_by_input: {path} must have input,output columns")
    x = np.asarray(data["input"], dtype=np.float32)
    y = np.asarray(data["output"], dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def compute_ulp_errors(activation, precision, inputs, outputs):
    add_tt_poly_fit_to_path()
    extract_accuracy = importlib.import_module("extract_accuracy")
    ground_truth = importlib.import_module("ground_truth")
    baseline = ground_truth.compute_ground_truth(activation, inputs)
    _, _, ulp_errors, valid = extract_accuracy.compute_error_arrays(
        baseline,
        outputs,
        precision=precision,
        inputs=inputs,
    )
    ulp_errors = np.where(valid, ulp_errors, np.nan)
    order = np.argsort(inputs)
    return inputs[order], ulp_errors[order]


def downsample(x, y, max_points):
    if not max_points or len(x) <= max_points:
        return x, y
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) <= max_points:
        return x, y
    keep_extrema = max(100, max_points // 10)
    extrema = np.argsort(y)[-keep_extrema:]
    remaining = max_points - len(extrema)
    regular = np.linspace(0, len(x) - 1, remaining, dtype=int) if remaining > 0 else np.array([], dtype=int)
    idx = np.unique(np.concatenate([extrema, regular]))
    idx.sort()
    return x[idx], y[idx]


def parse_series(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).stem
    return label, Path(path).expanduser().resolve()


def series_style(label, fallback):
    key = label.lower()
    if key == "ttnn":
        return "#D62728", "x"
    if key == "ours":
        return "#2CA02C", "^"
    if "min ulp" in key or "best ulp" in key or "lowest ulp" in key:
        return "#2CA02C", "^"
    if "tradeoff" in key or "knee" in key:
        return "#F28E2B", "s"
    if "fastest" in key or "min runtime" in key:
        return "#4E79A7", "o"
    return fallback, "o"


def series_draw_params(label):
    key = label.lower()
    if key == "ours":
        return 4.4, 0.86, 5
    if "min ulp" in key or "best ulp" in key or "lowest ulp" in key:
        return 4.2, 0.82, 5
    if "tradeoff" in key or "knee" in key:
        return 2.8, 0.72, 4
    if "fastest" in key or "min runtime" in key:
        return 1.8, 0.62, 3
    if key == "ttnn":
        return 1.2, 0.34, 2
    return 1.8, 0.62, 3


def add_low_ulp_inset(ax, series):
    low = [(label, x, y, color, marker) for label, x, y, color, marker in series if np.any((y >= 0) & (y <= 1.0))]
    if not low:
        return
    ymax = ax.get_ylim()[1]
    if ymax <= 2.0:
        return

    inset = ax.inset_axes([0.58, 0.52, 0.35, 0.30])
    inset.set_facecolor("#F7F7F7")
    for _, x, y, color, marker in low:
        mask = (y >= 0) & (y <= 1.0)
        size, alpha, zorder = series_draw_params(_)
        kwargs = {}
        if marker != "x":
            kwargs["edgecolors"] = "none"
        inset.scatter(
            x[mask],
            y[mask],
            s=max(1.0, size * 0.75),
            alpha=alpha,
            color=color,
            marker=marker,
            rasterized=True,
            zorder=zorder,
            **kwargs,
        )
    xmin, xmax = ax.get_xlim()
    inset.set_xlim(xmin, xmax)
    inset.set_ylim(0, 1.02)
    inset.set_title("0-1 ULP", fontsize=6.5, pad=1.5)
    inset.tick_params(axis="both", labelsize=6, length=2, pad=1)
    inset.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.35)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.45)
    inset.spines["bottom"].set_linewidth(0.45)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ULP error over input domain from raw dumps.")
    parser.add_argument("--activation", "-a", required=True)
    parser.add_argument("--precision", "-p", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--series", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--max-points", type=int, default=50000)
    return parser.parse_args()


def legend_handles(plt, series):
    from matplotlib.lines import Line2D

    handles = []
    seen = set()
    for label, _, _, color, marker in series:
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                label=label,
                markerfacecolor="none" if marker == "x" else color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=5.4,
                alpha=0.9,
            )
        )
    return handles


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_tufte_style(plt, compact=True)
    fig, ax = plt.subplots(figsize=(3.7, 2.35))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2"]
    plotted = []

    for i, spec in enumerate(args.series):
        label, path = parse_series(spec)
        if not path.exists():
            raise SystemExit(f"ulp_by_input: dump not found: {path}")
        inputs, outputs = load_dump(path)
        x, ulp = compute_ulp_errors(args.activation, args.precision, inputs, outputs)
        x, ulp = downsample(x, ulp, args.max_points)
        color, marker = series_style(label, colors[i % len(colors)])
        plotted.append((label, x, ulp, color, marker))
        size, alpha, zorder = series_draw_params(label)
        kwargs = {}
        if marker != "x":
            kwargs["edgecolors"] = "none"
        ax.scatter(
            x,
            ulp,
            s=size,
            alpha=alpha,
            color=color,
            label=label,
            marker=marker,
            rasterized=True,
            zorder=zorder,
            **kwargs,
        )

    ax.set_xlabel("Input domain")
    ax.set_ylabel(f"ULP error ({args.precision.upper()})")
    ax.set_ylim(bottom=0)
    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.04)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    ax.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.5)
    add_low_ulp_inset(ax, plotted)
    handles = legend_handles(plt, plotted)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=min(4, len(handles)),
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.95,
        borderaxespad=0.0,
    )
    fig.tight_layout()

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"# ULP-by-input plot -> {out}")


if __name__ == "__main__":
    main()
