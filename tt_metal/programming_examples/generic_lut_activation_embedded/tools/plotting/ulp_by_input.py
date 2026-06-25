#!/usr/bin/env python3
"""Plot ULP error versus input domain from raw hardware dump CSVs.

Input CSVs must contain columns:
  input,output

This intentionally accepts explicit dump paths instead of assuming the retired
generic_lut_activation/data/hardware_outputs layout.
"""

import argparse
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
    data = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ULP error over input domain from raw dumps.")
    parser.add_argument("--activation", "-a", required=True)
    parser.add_argument("--precision", "-p", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--series", action="append", required=True, metavar="LABEL=CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--max-points", type=int, default=50000)
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_tufte_style(plt)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2"]

    for i, spec in enumerate(args.series):
        label, path = parse_series(spec)
        if not path.exists():
            raise SystemExit(f"ulp_by_input: dump not found: {path}")
        inputs, outputs = load_dump(path)
        x, ulp = compute_ulp_errors(args.activation, args.precision, inputs, outputs)
        x, ulp = downsample(x, ulp, args.max_points)
        ax.scatter(
            x,
            ulp,
            s=2,
            alpha=0.45,
            edgecolors="none",
            color=colors[i % len(colors)],
            label=label,
            rasterized=True,
        )

    ax.set_xlabel("Input")
    ax.set_ylabel(f"ULP error ({args.precision.upper()})")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=min(4, len(args.series)), frameon=False)
    fig.tight_layout()

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"# ULP-by-input plot -> {out}")


if __name__ == "__main__":
    main()
