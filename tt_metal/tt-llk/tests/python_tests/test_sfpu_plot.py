# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Table-driven single-op SFPU accuracy harness with result plotting.

Each entry in CASES (near the bottom of this file) sweeps one SFPU op over an
input domain on hardware and emits a multi-panel accuracy plot + stats summary
(golden vs hardware: signed ULP error, relative error, ULP CDF, per-bin
percentiles, monotonicity).

To add a test, append a Case(...) to CASES and pick a format with
fmt=BF16/FP16/FP32 — see the "HOW TO ADD A TEST" comment above CASES. Run a
single op with:  pytest test_sfpu_plot.py -k <Op> -s
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.sfpu_domains import _SFPU_UNDEFINED_RANGES, Operand, _subtract_intervals
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import DistributionKind, StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)
from helpers.utils import passed_test

# ---------------------------------------------------------------------------
# House style for all plots produced by this file (module-level so it's
# applied once on import). Pure cosmetics — no impact on data or metrics.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "font.family": "DejaVu Sans",
    }
)


# ---------------------------------------------------------------------------
# Interval geometry helpers
# ---------------------------------------------------------------------------


def _intersect_segment(
    a: float,
    b: float,
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Parts of [a, b] that overlap any interval."""
    return [(max(a, lo), min(b, hi)) for lo, hi in intervals if max(a, lo) < min(b, hi)]


# Distributions whose sweep range is given by spec.low/spec.high. Others
# (gaussian / gaussian_linspace) are defined by mean/std, so their low/high stay
# at the 0/1 defaults — using those would falsely shade the rest as "excluded".
_LOW_HIGH_BOUNDED = {
    DistributionKind.UNIFORM,
    DistributionKind.RAMP,
    DistributionKind.SAW,
    DistributionKind.LOG_UNIFORM,
    DistributionKind.LOG_UNIFORM_LINSPACE,
    DistributionKind.SEQUENTIAL,
}


def _allowed_intervals_for(
    spec: StimuliSpec,
    x: np.ndarray,
) -> List[Tuple[float, float]]:
    """The x-regions that are the legitimate test domain, for plot shading.

    Explicit intervals win. For low/high-bounded distributions use [low, high].
    For mean/std-defined ones (gaussian) low/high are meaningless, so use the
    actual sampled span — otherwise everything outside [0, 1] is falsely shaded.
    """
    if spec.intervals:
        return spec.intervals
    if spec.distribution in _LOW_HIGH_BOUNDED:
        return [(spec.low, spec.high)]
    return [(float(x.min()), float(x.max()))]


# ---------------------------------------------------------------------------
# Binned percentile ULP analysis
# ---------------------------------------------------------------------------


def _compute_binned_ulp_stats(
    x: np.ndarray,
    signed_ulp_error: np.ndarray,
    num_bins: int = 32,
    x_for_bins: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Per-bin |ULP| statistics over a binning axis (defaults to x).

    Pass `x_for_bins=np.abs(x)`, `np.abs(y_golden)`, or
    `np.log10(np.abs(...))` later to bin over a different axis without
    changing the per-point ULP values themselves.

    Returned dict (each array has length `num_bins`, edges has +1):
        edges, centers, count, exact_frac, p50, p95, p99, max
    Empty bins → count=0 and NaN for the percentile / max values.
    """
    if x_for_bins is None:
        x_for_bins = x
    abs_ulp = np.abs(signed_ulp_error)
    valid = np.isfinite(x_for_bins) & np.isfinite(abs_ulp)

    nan_arr = np.full(num_bins, np.nan)
    if not valid.any():
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        return {
            "edges": edges,
            "centers": (edges[:-1] + edges[1:]) / 2,
            "count": np.zeros(num_bins, dtype=int),
            "exact_frac": nan_arr.copy(),
            "p50": nan_arr.copy(),
            "p95": nan_arr.copy(),
            "p99": nan_arr.copy(),
            "max": nan_arr.copy(),
        }

    xv = x_for_bins[valid]
    av = abs_ulp[valid]

    x_lo, x_hi = float(xv.min()), float(xv.max())
    if x_hi == x_lo:
        x_hi = x_lo + 1.0  # avoid zero-width bins on degenerate input
    edges = np.linspace(x_lo, x_hi, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # np.digitize returns 1..num_bins+1 — shift to 0..num_bins and clip the
    # right edge into the last bin.
    bin_idx = np.clip(np.digitize(xv, edges) - 1, 0, num_bins - 1)

    count = np.zeros(num_bins, dtype=int)
    exact_frac = nan_arr.copy()
    p50 = nan_arr.copy()
    p95 = nan_arr.copy()
    p99 = nan_arr.copy()
    max_ulp = nan_arr.copy()

    for b in range(num_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_data = av[mask]
        n = len(bin_data)
        count[b] = n
        exact_frac[b] = float((bin_data == 0).sum()) / n
        p50[b] = float(np.percentile(bin_data, 50))
        p95[b] = float(np.percentile(bin_data, 95))
        p99[b] = float(np.percentile(bin_data, 99))
        max_ulp[b] = float(bin_data.max())

    return {
        "edges": edges,
        "centers": centers,
        "count": count,
        "exact_frac": exact_frac,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max": max_ulp,
    }


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------
#
# Monotonicity is a DIFFERENT property from ULP accuracy:
#   - low ULP does not guarantee preserved ordering;
#   - an op can be tightly within 1 ULP and still occasionally invert two
#     adjacent outputs, which can break downstream sort/argmax logic.
# We use NON-strict comparison (equal neighbors are allowed) because bfloat16
# quantization legitimately collapses many adjacent inputs to the same output.

_MONOTONIC_OPS: Dict[MathOperation, str] = {
    # Strictly increasing on their full domain.
    MathOperation.Log: "increasing",
    MathOperation.Log1p: "increasing",
    MathOperation.Exp: "increasing",
    MathOperation.Exp2: "increasing",
    MathOperation.Sqrt: "increasing",
    MathOperation.Acosh: "increasing",
    MathOperation.Asinh: "increasing",
    MathOperation.Atanh: "increasing",
    MathOperation.Sigmoid: "increasing",
    MathOperation.Tanh: "increasing",
    MathOperation.Relu: "increasing",
    MathOperation.Rsqrt: "decreasing",
    MathOperation.Reciprocal: "decreasing",
}


def _monotonic_direction(op: MathOperation) -> Optional[str]:
    """Return 'increasing', 'decreasing', or None (op not registered)."""
    return _MONOTONIC_OPS.get(op)


def _check_monotonicity(
    x: np.ndarray,
    y_hw: np.ndarray,
    direction: str,
    allowed_intervals: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[int, List[Tuple[float, float, float, float, float]]]:
    """Adjacency-pair monotonicity check on sorted finite (x, y_hw).

    Splits the sequence into segments by `allowed_intervals` (each interval
    becomes its own segment) so we never compare across an excluded gap —
    crucial for ops like Reciprocal where each branch is monotonic on its
    own but the function jumps across 0.

    Equal neighbors are allowed (non-strict). For 'increasing', a violation
    is y_hw[i+1] < y_hw[i]; for 'decreasing', y_hw[i+1] > y_hw[i].

    Returns (n_pairs_checked, violations) where each violation is a tuple
    (x_left, x_right, y_left, y_right, |Δy|).
    """
    n = len(x)
    if n < 2:
        return 0, []

    # Build segment slices over the sorted x. With no allowed_intervals,
    # treat the whole array as a single segment.
    if allowed_intervals:
        segments: List[Tuple[int, int]] = []
        for lo, hi in sorted(allowed_intervals):
            mask = (x >= lo) & (x <= hi)
            if mask.any():
                idx = np.where(mask)[0]
                segments.append((int(idx[0]), int(idx[-1]) + 1))
    else:
        segments = [(0, n)]

    total_pairs = 0
    violations: List[Tuple[float, float, float, float, float]] = []
    for s_lo, s_hi in segments:
        if s_hi - s_lo < 2:
            continue
        seg_x = x[s_lo:s_hi]
        seg_y = y_hw[s_lo:s_hi]
        diffs = seg_y[1:] - seg_y[:-1]
        total_pairs += int(len(diffs))
        if direction == "increasing":
            # violation: y went DOWN going right
            viol_idx = np.where(diffs < 0)[0]
        else:  # "decreasing"
            # violation: y went UP going right
            viol_idx = np.where(diffs > 0)[0]
        for i in viol_idx:
            violations.append(
                (
                    float(seg_x[i]),
                    float(seg_x[i + 1]),
                    float(seg_y[i]),
                    float(seg_y[i + 1]),
                    float(abs(diffs[i])),
                )
            )
    return total_pairs, violations


# ---------------------------------------------------------------------------
# Shared plotting / stats
# ---------------------------------------------------------------------------

# ULP threshold staircase shared by the signed-error, relative-error, and CDF
# panels. "Show threshold i only if the data reached threshold i-1" so a clean
# op that never exceeds 3 ULP doesn't draw a 10-ULP line with no nearby data.
_ULP_THRESHOLDS = (
    (1, "#388e3c"),  # green
    (3, "#f57c00"),  # orange
    (10, "#d32f2f"),  # red
    (100, "#7b1fa2"),  # purple
    (1000, "#212121"),  # near-black (catastrophic outlier band)
)


def _visible_ulp_thresholds(max_val: float) -> List[Tuple[int, str]]:
    """(threshold, color) entries to draw given the data's max |ULP| = max_val.

    Always includes the first (1 ULP); includes threshold i (i>0) only if the
    data reached the previous threshold.
    """
    return [
        (t, c)
        for i, (t, c) in enumerate(_ULP_THRESHOLDS)
        if i == 0 or max_val >= _ULP_THRESHOLDS[i - 1][0]
    ]


def _plot_and_print(
    mathop: MathOperation,
    fmt: DataFormat,
    x: np.ndarray,
    y_golden: np.ndarray,
    y_hw: np.ndarray,
    plot_path: str,
    title_suffix: str = "",
    allowed_intervals: Optional[List[Tuple[float, float]]] = None,
    undefined_ranges: Optional[List[Tuple[float, float]]] = None,
):
    # Keep the raw inputs/outputs around so we can still surface non-finite
    # points on the top plot — even though they're masked out of error stats.
    x_raw = x.copy()
    y_golden_raw = y_golden.copy()
    y_hw_raw = y_hw.copy()

    # Drop points where either output is non-finite (inf/nan). These arise when
    # the op is undefined or overflows for an input (e.g. reciprocal of denormals).
    # inf - inf = nan would corrupt all downstream stats.
    finite_mask = np.isfinite(y_golden_raw) & np.isfinite(y_hw_raw)
    n_nonfinite = int((~finite_mask).sum())
    x, y_golden, y_hw = (
        x_raw[finite_mask],
        y_golden_raw[finite_mask],
        y_hw_raw[finite_mask],
    )

    # Classify non-finite points for marker overlay & summary breakdown.
    hw_nf = ~np.isfinite(y_hw_raw)
    golden_nf = ~np.isfinite(y_golden_raw)
    both_nf = hw_nf & golden_nf
    hw_only_nf = hw_nf & ~golden_nf
    gold_only_nf = golden_nf & ~hw_nf

    # Finer split: inf vs nan, on each side independently.
    hw_inf = np.isinf(y_hw_raw)
    hw_nan = np.isnan(y_hw_raw)
    gold_inf = np.isinf(y_golden_raw)
    gold_nan = np.isnan(y_golden_raw)

    # Every sampled output is non-finite -> nothing finite to plot or measure.
    # Emit a minimal figure with the inf/nan breakdown and return.
    if len(x) == 0:
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        msg = (
            f"All {n_nonfinite} sampled outputs are non-finite — nothing to plot.\n\n"
            f"both non-finite: {int(both_nf.sum())}   "
            f"HW-only: {int(hw_only_nf.sum())}   "
            f"golden-only: {int(gold_only_nf.sum())}\n"
            f"HW inf: {int(hw_inf.sum())}   HW nan: {int(hw_nan.sum())}   "
            f"golden inf: {int(gold_inf.sum())}   golden nan: {int(gold_nan.sum())}"
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(
            rf"SFPU {mathop.name} — {fmt.name}{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontfamily="monospace")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.warning(
            "[{}] all {} sampled outputs non-finite — minimal plot only",
            mathop.name,
            n_nonfinite,
        )
        return

    error = y_hw - y_golden
    # nonzero_mask guards only against division by zero in the relative error
    # calculation — do not use a large threshold like bfloat16.eps, which would
    # incorrectly exclude small-magnitude outputs (e.g. reciprocal of large inputs).
    nonzero_mask = y_golden != 0.0
    rel_error = np.where(nonzero_mask, np.abs(error) / np.abs(y_golden), 0.0)
    rel_error_valid = rel_error[nonzero_mask & (rel_error > 0)]
    bits_per_point = (
        -np.log2(rel_error_valid) if len(rel_error_valid) > 0 else np.array([])
    )

    # ULP scale for the relative-error reference lines on panel 3 (1/3/10/100
    # multiples of format eps). Note: the summary stats and panel 4 CDF do
    # NOT use this — they use the true local ULP derived from signed_ulp_error
    # below, so all reported ULP numbers stay consistent across the figure.
    if fmt == DataFormat.Float16_b:
        ulp_rel = float(torch.finfo(torch.bfloat16).eps)
    elif fmt == DataFormat.Float16:
        ulp_rel = float(torch.finfo(torch.float16).eps)
    elif fmt == DataFormat.Float32:
        ulp_rel = float(torch.finfo(torch.float32).eps)
    else:
        ulp_rel = None

    # Signed ULP error for every finite point. Uses TRUE local ULP via
    # torch.nextafter on the golden output values cast to the target format —
    # this avoids the near-zero blow-up of the previous approximate
    # normalization (error / (|y_golden| * eps)) where tiny |y_golden|
    # made trivial absolute errors look like enormous ULPs.
    # 0 wherever y_golden == 0 (ULP at zero collapses to a sub-normal that
    # would re-introduce the same blow-up) or where the local ULP is 0.
    # Used by axes[1] (stem plot) and axes[3] (CDF).
    if fmt in (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32):
        torch_dtype = format_dict[fmt]
        abs_golden_t = torch.tensor(np.abs(y_golden), dtype=torch_dtype)
        next_up_t = torch.nextafter(
            abs_golden_t,
            torch.tensor(float("inf"), dtype=torch_dtype),
        )
        local_ulp = (next_up_t - abs_golden_t).to(torch.float32).numpy()
        valid_local = (y_golden != 0.0) & (local_ulp > 0)
        safe_local_ulp = np.where(valid_local, local_ulp, 1.0)
        signed_ulp_error = np.where(valid_local, error / safe_local_ulp, 0.0)
    else:
        signed_ulp_error = None

    # ULP magnitude per non-zero-error point — derived from signed_ulp_error so
    # the summary stats (Mean/p99/Max ULP, "Points > N ULP", top offenders)
    # use the SAME true-local-ULP definition as panels 2 and 4. ulp_err_mask
    # selects exactly those rows for the offenders table x/golden/hw lookups.
    if signed_ulp_error is not None:
        abs_signed_ulp = np.abs(signed_ulp_error)
        ulp_err_mask = abs_signed_ulp > 0
        ulp_err = abs_signed_ulp[ulp_err_mask]
    else:
        ulp_err_mask = None
        ulp_err = np.array([])

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    # Two-column layout: 5 plot panels stacked on the left, one tall text-summary
    # panel on the right. Existing code keeps axes[0]..axes[4] indexing; the
    # new binned-percentile panel lives at axes[5].
    fig, ax_dict = plt.subplot_mosaic(
        [
            ["top", "summary"],
            ["err", "summary"],
            ["rel", "summary"],
            ["hist", "summary"],
            ["binned", "summary"],
        ],
        figsize=(20, 17),
        gridspec_kw={
            "width_ratios": [3, 1.5],
            # err panel (signed ULP stem) gets more vertical room so the
            # individual stems are readable rather than crammed.
            "height_ratios": [3, 4, 3, 3, 3],
        },
    )
    fig.patch.set_facecolor("#fafafa")
    axes = [
        ax_dict["top"],
        ax_dict["err"],
        ax_dict["rel"],
        ax_dict["hist"],
        ax_dict["summary"],
        ax_dict["binned"],
    ]
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    axes[5].sharex(axes[0])

    GOLDEN_COLOR = "#0d47a1"  # deep blue
    HW_COLOR = "#ff7f0e"  # orange

    # Plot lines per allowed-interval segment so they don't bridge across the
    # shaded undefined / excluded regions. Scatter still draws every sampled
    # point regardless of intervals.
    line_segments = (
        sorted(allowed_intervals) if allowed_intervals else [(x.min(), x.max())]
    )
    first_g = first_h = True
    for lo, hi in line_segments:
        seg_mask = (x >= lo) & (x <= hi)
        if not seg_mask.any():
            continue
        axes[0].plot(
            x[seg_mask],
            y_golden[seg_mask],
            label="Golden (torch)" if first_g else "_nolegend_",
            linewidth=1.0,
            color=GOLDEN_COLOR,
            alpha=0.7,
            zorder=2,
        )
        first_g = False
        axes[0].plot(
            x[seg_mask],
            y_hw[seg_mask],
            label="Hardware" if first_h else "_nolegend_",
            linewidth=1.0,
            color=HW_COLOR,
            alpha=0.7,
            linestyle="--",
            zorder=3,
        )
        first_h = False
    axes[0].scatter(x, y_golden, s=8, alpha=0.7, color=GOLDEN_COLOR, zorder=4)
    axes[0].scatter(
        x,
        y_hw,
        s=18,
        alpha=0.7,
        facecolors="none",
        edgecolors=HW_COLOR,
        linewidths=0.5,
        zorder=5,
    )

    # Visual-only reference lines at the asymptotes of atanh(x) at x = ±1.
    # Does not affect sampling, allowed_intervals, or finite-mask behavior.
    if mathop == MathOperation.Atanh:
        axes[0].axvline(-1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
        axes[0].axvline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)

    # Shade excluded / undefined regions from explicit interval specs.
    def _fmt_interval(lo: float, hi: float) -> str:
        def _f(v: float) -> str:
            if v == float("inf"):
                return "+inf"
            if v == -float("inf"):
                return "-inf"
            return f"{v:g}"

        return f"[{_f(lo)}, {_f(hi)}]"

    def _fmt_intervals(intervals: List[Tuple[float, float]]) -> str:
        return " ∪ ".join(_fmt_interval(lo, hi) for lo, hi in intervals)

    if allowed_intervals is not None and len(x):
        x_min, x_max = float(x.min()), float(x.max())
        complement = _subtract_intervals([(x_min, x_max)], allowed_intervals)
        # Collect the actual visible ranges so the legend tells the user which
        # x-spans are shaded, not just the registry/spec inputs.
        all_undef_parts: List[Tuple[float, float]] = []
        all_excl_parts: List[Tuple[float, float]] = []
        for seg_lo, seg_hi in complement:
            up = _intersect_segment(seg_lo, seg_hi, undefined_ranges or [])
            ep = _subtract_intervals([(seg_lo, seg_hi)], up)
            all_undef_parts.extend(up)
            all_excl_parts.extend(ep)

        undef_label = (
            f"Undefined domain {_fmt_intervals(all_undef_parts)}"
            if all_undef_parts
            else None
        )
        excl_label = (
            f"Excluded by intervals {_fmt_intervals(all_excl_parts)}"
            if all_excl_parts
            else None
        )
        # Shade on all three input-x panels so the regions are visible in the
        # function plot, the signed-error plot, and the relative-error plot.
        # Only axes[0]'s entries get a legend label; the others reuse the same
        # geometry without polluting their own legends.
        x_panels = (axes[0], axes[1], axes[2])
        undef_seen = excl_seen = False
        for ulo, uhi in all_undef_parts:
            top_lbl = "_nolegend_" if undef_seen else undef_label
            x_panels[0].axvspan(
                ulo, uhi, alpha=0.15, color="red", label=top_lbl, zorder=0
            )
            for ax in x_panels[1:]:
                ax.axvspan(ulo, uhi, alpha=0.15, color="red", zorder=0)
            undef_seen = True
        for elo, ehi in all_excl_parts:
            top_lbl = "_nolegend_" if excl_seen else excl_label
            x_panels[0].axvspan(
                elo, ehi, alpha=0.25, color="#808080", label=top_lbl, zorder=0
            )
            for ax in x_panels[1:]:
                ax.axvspan(elo, ehi, alpha=0.25, color="#808080", zorder=0)
            excl_seen = True

    # Overlay markers for non-finite outputs at the top of the plot. Three
    # tiers: both inputs nf (red), HW-only nf (orange), golden-only nf (blue).
    if hw_nf.any() or golden_nf.any():
        y_min, y_max = axes[0].get_ylim()
        y_span = y_max - y_min if (y_max > y_min) else 1.0
        for mask, color, offset_frac, lbl, size in (
            (both_nf, "red", 0.00, "inf/nan (both)", 40),
            (hw_only_nf, "orange", 0.05, "inf/nan (HW only)", 35),
            (gold_only_nf, "C0", 0.10, "inf/nan (golden only)", 35),
        ):
            xs = x_raw[mask]
            if len(xs):
                ys = np.full_like(xs, y_max - offset_frac * y_span)
                axes[0].scatter(
                    xs,
                    ys,
                    marker="v",
                    s=size,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    label=lbl,
                    zorder=6,
                )

    op_to_formula = {
        MathOperation.Reciprocal: r"$1/x$",
        MathOperation.Sqrt: r"$\sqrt{x}$",
        MathOperation.Rsqrt: r"$1/\sqrt{x}$",
        MathOperation.Exp: r"$e^x$",
        MathOperation.Exp2: r"$2^x$",
        MathOperation.Log: r"$\log(x)$",
        MathOperation.Log1p: r"$\log(1+x)$",
        MathOperation.Atanh: r"$\tanh^{-1}(x)$",
        MathOperation.Asinh: r"$\sinh^{-1}(x)$",
        MathOperation.Acosh: r"$\cosh^{-1}(x)$",
        MathOperation.Sin: r"$\sin(x)$",
        MathOperation.Cos: r"$\cos(x)$",
        MathOperation.Tanh: r"$\tanh(x)$",
        MathOperation.Sigmoid: r"$\sigma(x)$",
        MathOperation.Silu: r"$x\,\sigma(x)$",
        MathOperation.Gelu: r"$\mathrm{gelu}(x)$",
        MathOperation.Elu: r"$\mathrm{elu}(x)$",
        MathOperation.Celu: r"$\mathrm{celu}(x)$",
        MathOperation.Hardsigmoid: r"$\mathrm{hardsigmoid}(x)$",
        MathOperation.Relu: r"$\max(0, x)$",
        MathOperation.Square: r"$x^2$",
        MathOperation.Abs: r"$|x|$",
        MathOperation.Neg: r"$-x$",
    }
    formula = op_to_formula.get(mathop, "")
    axes[0].set_ylabel(f"{mathop.name}(x)")
    fig.suptitle(
        rf"SFPU {mathop.name} ({formula}) — {fmt.name}{title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    subtitle = f"x ∈ [{x.min():.2g}, {x.max():.2g}]"
    if n_nonfinite:
        subtitle += f"  ({n_nonfinite} inf/nan excluded)"
    axes[0].set_title(subtitle, fontsize=9, color="#666666")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convert signed error to ULPs when the format has a known ULP. ULP units
    # normalize across the input range — a "small" absolute error near a large
    # output may be many ULPs while a "large" absolute error near a small
    # output may be sub-ULP. Consistent positive or negative stems = systematic
    # approximation bias. For non-16-bit formats, fall back to absolute error.
    if signed_ulp_error is not None:
        err_for_panel = signed_ulp_error
        err_ylabel = "Signed error [ULP]"
        err_title = "Signed error (hw - golden), in ULPs"
    else:
        err_for_panel = error
        err_ylabel = "Error (hw - golden) [absolute]"
        err_title = "Signed error (hw - golden)"

    # Stem plot makes the sign and magnitude of each point obvious — kept thin
    # and semi-transparent so dense data doesn't turn it into a solid block.
    markerline, stemlines, baseline = axes[1].stem(
        x,
        err_for_panel,
        linefmt="-",
        markerfmt="o",
        basefmt=" ",
    )
    plt.setp(stemlines, color="red", alpha=0.35, linewidth=0.5)
    plt.setp(markerline, color="red", alpha=0.6, markersize=2)
    axes[1].set_ylabel(err_ylabel)
    axes[1].set_title(err_title)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    if ulp_rel is not None:
        max_pos = float(max(0.0, signed_ulp_error.max()))
        max_neg = float(max(0.0, -signed_ulp_error.min()))  # |most-negative|
        # Two-sided: apply the shared staircase rule independently per side.
        pos_visible = {t for t, _ in _visible_ulp_thresholds(max_pos)}
        neg_visible = {t for t, _ in _visible_ulp_thresholds(max_neg)}
        legend_handles = []
        for mult, color in _ULP_THRESHOLDS:
            show_pos = mult in pos_visible
            show_neg = mult in neg_visible
            if show_pos:
                axes[1].axhline(
                    y=mult,
                    color=color,
                    linewidth=0.7,
                    linestyle=(0, (6, 3)),
                    alpha=0.5,
                )
            if show_neg:
                axes[1].axhline(
                    y=-mult,
                    color=color,
                    linewidth=0.7,
                    linestyle=(0, (6, 3)),
                    alpha=0.5,
                )
            if show_pos and show_neg:
                label = f"±{mult} ULP"
            elif show_pos:
                label = f"+{mult} ULP"
            elif show_neg:
                label = f"−{mult} ULP"
            else:
                continue
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=1.6,
                    # Exactly 2 same-sized dashes per swatch: dash 4pt, gap 4pt
                    # cycled into a handle that's ~12pt wide at handlelength=1.5.
                    linestyle=(0, (4, 4)),
                    alpha=1.0,
                    label=label,
                )
            )
        if legend_handles:
            axes[1].legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=8,
                ncol=2,
                handlelength=1.5,
            )

    # Only plot points with non-zero relative error: log scale can't represent 0
    # (points where hw == golden exactly are silently dropped by matplotlib).
    plot_mask = nonzero_mask & (rel_error > 0)
    n_exact = int(nonzero_mask.sum()) - int(plot_mask.sum())
    if plot_mask.any():
        axes[2].scatter(
            x[plot_mask], rel_error[plot_mask], s=1, alpha=0.5, color="blue"
        )
    if ulp_rel is not None:
        max_ulp_rel2 = (
            float(rel_error_valid.max() / ulp_rel) if len(rel_error_valid) > 0 else 0.0
        )
        for mult, color in _visible_ulp_thresholds(max_ulp_rel2):
            axes[2].axhline(
                mult * ulp_rel,
                color=color,
                linewidth=1.2,
                linestyle=(0, (6, 3)),
                alpha=0.85,
                label=f"{mult} ULP",
            )
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("Input value")
    axes[2].set_ylabel("Relative error")
    axes[2].set_title("Relative error |hw - golden| / |golden|")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    # CDF of |ULP error| — directly answers "what fraction of points are
    # within N ULPs?" with explicit annotations at standard thresholds.
    # For non-16-bit formats (no defined ULP), fall back to a relative-error
    # histogram with log counts.
    if signed_ulp_error is not None:
        valid_for_cdf = y_golden != 0.0
        if valid_for_cdf.any():
            ulp_err_mag = np.abs(signed_ulp_error[valid_for_cdf])
            sorted_ulp = np.sort(ulp_err_mag)
            n = len(sorted_ulp)
            cdf = np.arange(1, n + 1) / n
            axes[3].plot(sorted_ulp, cdf, color="#0d47a1", linewidth=1.5)
            axes[3].set_xscale("log")
            max_ulp = float(sorted_ulp.max())
            visible_thresholds = _visible_ulp_thresholds(max_ulp)
            for threshold, color in visible_thresholds:
                frac = float(np.searchsorted(sorted_ulp, threshold, side="right")) / n
                axes[3].axvline(
                    threshold,
                    color=color,
                    linestyle=(0, (5, 3)),
                    linewidth=0.8,
                    alpha=0.8,
                )
                axes[3].text(
                    threshold,
                    frac,
                    f"  {threshold:g} ULP: {frac:.1%}",
                    fontsize=8,
                    va="center",
                    ha="left",
                    color=color,
                )
            # Cap x at slightly past the highest visible threshold or the
            # actual data max — whichever is larger. Left bound stays auto so
            # the smallest positive ULP still gets framed by log-x.
            if visible_thresholds:
                last_t = visible_thresholds[-1][0]
                x_max = max(last_t * 1.3, max_ulp * 1.1)
                axes[3].set_xlim(right=x_max)
            # Zoom y in to start at the FIRST VISIBLE CDF point on the log-x
            # axis — i.e. the CDF value at the first strictly positive ULP,
            # not cdf[0] (which corresponds to a zero x that log-x can't show).
            # Padding scales with how much CDF range is left above that point:
            # high first-visible y (clean op like Reciprocal) → small pad;
            # lower first-visible y (noisier op like Exp) → larger pad.
            positive_mask = sorted_ulp > 0
            if positive_mask.any():
                positive_idx = int(np.argmax(positive_mask))
                first_visible_y = float(cdf[positive_idx])
                visible_span = max(1e-4, 1.0 - first_visible_y)
                pad = float(np.clip(0.25 * visible_span, 0.001, 0.02))
                y_lo = max(0.0, first_visible_y - pad)
                axes[3].set_ylim(y_lo, 1.005)
            else:
                axes[3].set_ylim(0, 1.005)

            # Make the bit-exact fraction explicit since it's not visually
            # representable on a log-x axis (log(0) is undefined). Placed in
            # the top-left corner with a small white bbox so it stays legible
            # even if the CDF curve passes nearby.
            zero_frac = float((ulp_err_mag == 0).sum()) / len(ulp_err_mag)
            axes[3].text(
                0.02,
                0.97,
                f"0 ULP exact matches: {zero_frac:.1%}",
                transform=axes[3].transAxes,
                fontsize=9,
                color="#444444",
                ha="left",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#cccccc",
                    linewidth=0.6,
                    alpha=0.9,
                ),
            )
        else:
            axes[3].text(
                0.5,
                0.5,
                "No nonzero-golden points",
                transform=axes[3].transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        axes[3].set_xlabel("ULP error")
        axes[3].set_ylabel("Fraction of points ≤ threshold")
        axes[3].set_title("CDF of |ULP error| (zoomed y-axis)")
    elif len(rel_error_valid) > 0:
        axes[3].hist(
            rel_error_valid,
            bins=50,
            log=True,
            color="#0d47a1",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[3].set_xlabel("Relative error")
        axes[3].set_ylabel("Count (log)")
        axes[3].set_title("Relative error distribution")
    else:
        axes[3].text(
            0.5,
            0.5,
            "No data to plot",
            transform=axes[3].transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        axes[3].set_xlabel("|ULP error|")
        axes[3].set_ylabel("Fraction of points within")
        axes[3].set_title("CDF of |ULP error|")
    axes[3].grid(True, alpha=0.3)

    # Binned percentile ULP — uniform bins over x. Helps spot input regions
    # where the SFPU is systematically worse vs random rounding noise. Reuses
    # the same true-local ULP from signed_ulp_error so the bin stats match the
    # numbers reported in the summary box and panels 2 / 4.
    #
    # INTERPRETATION CAVEAT: ULP magnitude is normalized by the local spacing
    # of the OUTPUT format. For ops where the output is near 0 (e.g. log(x)
    # at x ≈ 1), local ULP is tiny and a small absolute error blows up to a
    # large ULP count. Big ULP spikes there are usually a normalization
    # artifact, not a catastrophic absolute error.
    #
    # FUTURE: the helper accepts `x_for_bins` so a magnitude-based view —
    # bin by |y_golden| or log10(|y_golden|) — can be added without changing
    # the helper signature.
    MIN_BIN_COUNT = 8  # bins below this are too small for reliable percentiles
    if signed_ulp_error is not None:
        binned = _compute_binned_ulp_stats(x, signed_ulp_error, num_bins=32)
        non_empty = binned["count"] > 0
        # Suppress percentile lines for thinly-populated bins so a 1-sample
        # outlier can't pull p99/max into a misleading peak.
        trustworthy = non_empty & (binned["count"] >= MIN_BIN_COUNT)
        if trustworthy.any():
            c = binned["centers"][trustworthy]
            for stat_name, color, marker in (
                ("p50", "#388e3c", "o"),
                ("p95", "#f57c00", "s"),
                ("p99", "#d32f2f", "^"),
                ("max", "#7b1fa2", "v"),
            ):
                axes[5].plot(
                    c,
                    binned[stat_name][trustworthy],
                    color=color,
                    linewidth=1.3,
                    alpha=0.85,
                    marker=marker,
                    markersize=3.5,
                    label=stat_name,
                )
            # symlog keeps tiny / zero values readable while still letting
            # multi-ULP outliers in the same view.
            axes[5].set_yscale("symlog", linthresh=0.5)
            axes[5].legend(loc="upper right", fontsize=8, ncol=4)
            # Secondary axis: per-bin exact-match fraction as a faint blue
            # filled step so it gives context without competing with the
            # percentile lines. Drawn for ALL non-empty bins (a fraction is
            # meaningful even at small N).
            ax5b = axes[5].twinx()
            ax5b.fill_between(
                binned["centers"][non_empty],
                0.0,
                binned["exact_frac"][non_empty],
                color="#1976d2",
                alpha=0.12,
                step="mid",
            )
            ax5b.set_ylim(0, 1.02)
            ax5b.set_ylabel("Exact-match fraction", color="#1976d2", fontsize=9)
            ax5b.tick_params(axis="y", labelcolor="#1976d2", labelsize=8)
            # Note on the count threshold so readers know why some bins
            # are missing from the percentile lines.
            n_skipped = int((non_empty & ~trustworthy).sum())
            note = f"percentiles: bins with count ≥ {MIN_BIN_COUNT} only"
            if n_skipped:
                note += (
                    f"  ({n_skipped} thin bin{'s' if n_skipped != 1 else ''} hidden)"
                )
            axes[5].text(
                0.02,
                0.97,
                note,
                transform=axes[5].transAxes,
                fontsize=8,
                color="#666666",
                ha="left",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#cccccc",
                    linewidth=0.6,
                    alpha=0.85,
                ),
            )
        elif non_empty.any():
            # Data exists but every bin is too thin — say so explicitly.
            axes[5].text(
                0.5,
                0.5,
                f"All bins have < {MIN_BIN_COUNT} samples — increase point count",
                ha="center",
                va="center",
                transform=axes[5].transAxes,
                fontsize=10,
                color="gray",
            )
        else:
            axes[5].text(
                0.5,
                0.5,
                "No valid binned data",
                ha="center",
                va="center",
                transform=axes[5].transAxes,
                fontsize=10,
                color="gray",
            )
        axes[5].set_xlabel("Input value (bin)")
        axes[5].set_ylabel("|ULP error|")
        axes[5].set_title("Per-bin |ULP|: p50 / p95 / p99 / max over input")
    else:
        axes[5].text(
            0.5,
            0.5,
            "Binned ULP available only for float formats (Float16_b / Float16 / Float32)",
            ha="center",
            va="center",
            transform=axes[5].transAxes,
            fontsize=10,
            color="gray",
        )
        axes[5].set_title("Per-bin |ULP|")
    axes[5].grid(True, alpha=0.3)

    # Build the textual summary once — used both in the figure (axes[4]) and
    # printed to the console. Anything inf/nan-related is collected separately
    # so it can be rendered as its own box on the right-hand panel.
    summary_lines: List[str] = []
    summary_lines.append(
        f"Input range: [{x.min():.4f}, {x.max():.4f}], {len(x)} finite points"
    )

    nf_detail_lines: List[str] = []
    if n_nonfinite > 0:
        nf_detail_lines.append(f"Non-finite outputs (inf/nan): {n_nonfinite}")
        nf_detail_lines.append(
            f"  both non-finite:   {int(both_nf.sum())}, "
            f"HW-only: {int(hw_only_nf.sum())}, "
            f"golden-only: {int(gold_only_nf.sum())}"
        )
        nf_detail_lines.append(
            f"  HW inf: {int(hw_inf.sum())}, HW nan: {int(hw_nan.sum())}, "
            f"golden inf: {int(gold_inf.sum())}, golden nan: {int(gold_nan.sum())}"
        )
        nf_idx = np.where(~finite_mask)[0]
        max_list = min(10, len(nf_idx))
        nf_detail_lines.append("")
        nf_detail_lines.append(f"Non-finite detail (up to {max_list} points):")
        nf_detail_lines.append(f"  {'x':>14}  {'golden':>14}  {'hw':>14}  {'type':>12}")
        for i in nf_idx[:max_list]:
            x_val = x_raw[i]
            g_val = y_golden_raw[i]
            h_val = y_hw_raw[i]
            if np.isnan(g_val) or np.isnan(h_val):
                kind = "nan"
            elif np.isinf(g_val) or np.isinf(h_val):
                kind = "inf"
            else:
                kind = "nf"
            nf_detail_lines.append(
                f"  {x_val:>14.6e}  {g_val:>14.6e}  {h_val:>14.6e}  {kind:>12}"
            )
    if n_exact > 0:
        summary_lines.append(
            f"{n_exact} point{'s' if n_exact != 1 else ''} with rel error = 0 "
            f"(hw == golden exactly, not shown on log scale)"
        )
    summary_lines.append(f"Max absolute error:  {np.abs(error).max():.6e}")
    summary_lines.append(f"Mean absolute error: {np.abs(error).mean():.6e}")
    if len(rel_error_valid) > 0:
        summary_lines.append(f"Max relative error:  {rel_error_valid.max():.2e}")
        summary_lines.append(f"Median rel error:    {np.median(rel_error_valid):.2e}")
        summary_lines.append(f"Bits of precision (worst):  {bits_per_point.min():.1f}")
        summary_lines.append(
            f"Bits of precision (median): {np.median(bits_per_point):.1f}"
        )
    else:
        summary_lines.append("No nonzero errors — hw matches golden exactly")

    # ULP statistics + top offenders (16-bit float formats only).
    if ulp_rel is not None and len(ulp_err) > 0:
        summary_lines.append("")
        summary_lines.append(f"ULP unit (eps):      {ulp_rel:.3e}")
        summary_lines.append(f"Mean ULP:            {ulp_err.mean():.2f}")
        summary_lines.append(f"p99 ULP:             {np.percentile(ulp_err, 99):.2f}")
        summary_lines.append(f"Max ULP:             {ulp_err.max():.2f}")
        summary_lines.append(
            f"Points > 3 ULP:      {int((ulp_err > 3).sum())} / {len(ulp_err)}"
        )
        summary_lines.append(
            f"Points > 10 ULP:     {int((ulp_err > 10).sum())} / {len(ulp_err)}"
        )
        summary_lines.append(
            f"Points > 100 ULP:    {int((ulp_err > 100).sum())} / {len(ulp_err)}"
        )
        summary_lines.append(
            f"Points > 1000 ULP:   {int((ulp_err > 1000).sum())} / {len(ulp_err)}"
        )

        # ulp_err_mask selects the same rows that ulp_err was built from,
        # so the indexing into x/golden/hw stays aligned.
        x_off = x[ulp_err_mask]
        g_off = y_golden[ulp_err_mask]
        h_off = y_hw[ulp_err_mask]
        top_n = min(10, len(ulp_err))
        top_idx = np.argsort(-ulp_err)[:top_n]
        summary_lines.append("")
        summary_lines.append(f"Top {top_n} ULP offenders:")
        summary_lines.append(f"  {'x':>14}  {'golden':>14}  {'hw':>14}  {'ulp':>10}")
        for i in top_idx:
            summary_lines.append(
                f"  {x_off[i]:>14.6e}  {g_off[i]:>14.6e}  {h_off[i]:>14.6e}  {ulp_err[i]:>10.2f}"
            )

    # Monotonicity check (independent of ULP — a pair can be sub-ULP and
    # still violate ordering, which can break downstream sort / argmax).
    # For unregistered ops the section is omitted.
    direction = _monotonic_direction(mathop)
    if direction is not None:
        n_pairs, mono_violations = _check_monotonicity(
            x,
            y_hw,
            direction,
            allowed_intervals,
        )
        summary_lines.append("")
        summary_lines.append(f"Monotonicity ({direction}):")
        summary_lines.append(f"  Pairs checked: {n_pairs}")
        summary_lines.append(f"  Violations:    {len(mono_violations)}")
        if n_pairs > 0:
            rate = len(mono_violations) / n_pairs
            summary_lines.append(f"  Rate:          {rate:.2%}")
        if mono_violations:
            mono_violations.sort(key=lambda v: -v[4])  # by |Δy| descending
            worst = mono_violations[0]
            summary_lines.append(
                f"  Worst |Δy|:    {worst[4]:.3e}  at x∈[{worst[0]:.3e}, {worst[1]:.3e}]"
            )
            top_n_viol = min(5, len(mono_violations))
            summary_lines.append(f"  Top {top_n_viol} violations:")
            summary_lines.append(
                f"  {'x1':>12}  {'x2':>12}  {'y1_hw':>14}  {'y2_hw':>14}  {'|Δy|':>10}"
            )
            for x1, x2, y1, y2, mag in mono_violations[:top_n_viol]:
                summary_lines.append(
                    f"  {x1:>12.4e}  {x2:>12.4e}  {y1:>14.6e}  {y2:>14.6e}  {mag:>10.3e}"
                )

    # Render the summary on the right-side panel as two stacked styled boxes:
    # main stats near the top, non-finite detail table below it.
    axes[4].axis("off")
    summary_bbox_kwargs = dict(
        boxstyle="round,pad=0.8",
        facecolor="#f5f5f5",
        edgecolor="#888888",
        linewidth=1.0,
    )
    axes[4].text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        transform=axes[4].transAxes,
        fontfamily="monospace",
        fontsize=11,
        va="top",
        ha="left",
        bbox=summary_bbox_kwargs,
    )
    if nf_detail_lines:
        axes[4].text(
            0.02,
            0.40,
            "\n".join(nf_detail_lines),
            transform=axes[4].transAxes,
            fontfamily="monospace",
            fontsize=11,
            va="top",
            ha="left",
            bbox=summary_bbox_kwargs,
        )

    plt.tight_layout()

    # Draw a thin vertical separator between the plot column and the summary
    # column. Done after tight_layout so the position lines up with the actual
    # column boundary.
    summary_bbox = axes[4].get_position()
    sep_x = summary_bbox.x0 - 0.005
    fig.add_artist(
        plt.Line2D(
            [sep_x, sep_x],
            [0.04, 0.96],
            transform=fig.transFigure,
            color="#bbbbbb",
            linewidth=1.2,
            alpha=0.8,
        )
    )

    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info("Plot saved to: {}", os.path.abspath(plot_path))
    logger.info("\n{}", "\n".join(summary_lines + nf_detail_lines))


# ---------------------------------------------------------------------------
# Stress harness — table-driven SFPU accuracy plots
# ---------------------------------------------------------------------------
#
# HOW TO ADD A TEST
#   Append one Case(...) to the CASES list below — that is the whole workflow:
#
#       Case(op=MathOperation.Exp, spec=StimuliSpec.ramp(low=-10.0, high=10.0))
#
#   Choose the input domain with a StimuliSpec (ramp / uniform / ... — the same
#   API the rest of the suite uses) and the numeric format with fmt=:
#
#       fmt=BF16   bfloat16 in/out   (default; judge accuracy in ULPs)
#       fmt=FP16   float16  in/out
#       fmt=FP32   float32  in/out   (auto-runs the fp32 dest-accumulator path;
#                                     judge accuracy in abs/rel error — fp32 ULP
#                                     is not meaningful, see the notes inside
#                                     _plot_and_print)
#
#   Run one op:   pytest test_sfpu_plot.py -k Exp -s
#   Run all:      pytest test_sfpu_plot.py -s
#
#   Each case writes _plot_output/sfpu_<id>.png and prints a stats summary, then
#   asserts the hardware result matches golden.
#   Set expect_pass=False to keep the run green while exploring a known-inaccurate op.
#
# Optional Case fields (advanced — sensible defaults cover the common cases):
#   StimuliSpec(intervals=[(lo, hi), ...])  sweep disjoint bands (e.g. either
#                                           side of a singularity)
#   extra_undefined_ranges  override the red "undefined domain" plot shading
#   clamp_negative          enable the kernel's negative-input clamp
#   dest_acc / unpack_to_dest  override the format-derived accumulator defaults
#   name                    custom test id (name of file) (defaults to "<Op>-<fmt>")

BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
FP16 = InputOutputFormat(DataFormat.Float16, DataFormat.Float16)
FP32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

_FMT_SHORT = {
    DataFormat.Float16_b: "bf16",
    DataFormat.Float16: "fp16",
    DataFormat.Float32: "fp32",
}


@dataclass
class Case:
    """One SFPU stress configuration. Add a Case to CASES to add a test."""

    op: MathOperation
    spec: StimuliSpec
    fmt: InputOutputFormat = BF16
    expect_pass: bool = True
    name: Optional[str] = None
    # Advanced overrides — defaults are derived from `fmt`.
    dest_acc: Optional[DestAccumulation] = None
    unpack_to_dest: Optional[bool] = None
    clamp_negative: bool = False
    input_dimensions: Optional[List[int]] = None
    extra_undefined_ranges: Optional[List[Tuple[float, float]]] = None

    @property
    def test_id(self) -> str:
        if self.name:
            return self.name
        short = _FMT_SHORT.get(self.fmt.output_format, self.fmt.output_format.name)
        return f"{self.op.name}-{short}"


# ---------------------------------------------------------------------------
# CASES — add rows here. The two below are templates: copy one, tweak, done.
# ---------------------------------------------------------------------------
CASES = [
    # bfloat16 demo: log over a positive (in-domain) range.
    Case(op=MathOperation.Log, spec=StimuliSpec.ramp(low=0.5, high=10.0)),
    # float32 demo: sqrt over [0, 100] (auto fp32 dest-accumulator path).
    Case(op=MathOperation.Sqrt, spec=StimuliSpec.ramp(low=0.0, high=100.0), fmt=FP32),
    # intervals demo: 1/x sampled on both sides of 0 but never on the
    # singularity at 0 (the gap between the two bands is skipped).
    Case(
        op=MathOperation.Reciprocal,
        spec=StimuliSpec.uniform(intervals=[(-10.0, -0.01), (0.01, 10.0)]),
    ),
    # Diagnostic-only example (uncomment to explore a known-inaccurate op without failing the run):
    # Case(op=MathOperation.Gelu, spec=StimuliSpec.ramp(low=-13.0, high=13.0), expect_pass=False),
]


def run_case(case: Case) -> bool:
    """Run one Case end-to-end: stimuli -> golden -> hardware -> plot + stats.

    Returns whether the hardware result matched golden (passed_test) and writes
    _plot_output/sfpu_<id>.png. This only assembles inputs for _plot_and_print;
    it does not alter any metric or plotting behavior.
    """
    formats = case.fmt
    is_fp32 = formats.output_format == DataFormat.Float32

    # Format-derived defaults: the fp32 path needs a real fp32 dest accumulator
    # (dest_acc=Yes + unpack_to_dest=True) so the SFPU body takes its fp32
    # branch. Both stay overridable on the Case.
    dest_acc = case.dest_acc
    if dest_acc is None:
        dest_acc = DestAccumulation.Yes if is_fp32 else DestAccumulation.No
    unpack_to_dest = case.unpack_to_dest
    if unpack_to_dest is None:
        unpack_to_dest = is_fp32

    input_dimensions = case.input_dimensions or [32, 32]
    mathop = case.op
    spec = case.spec
    plot_path = f"_plot_output/sfpu_{case.test_id}.png"

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        spec_A=spec,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(case.clamp_negative),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    sort_idx = torch.argsort(src_A.to(torch.float32))
    x = src_A.to(torch.float32)[sort_idx].numpy()
    y_golden = golden_tensor.to(torch.float32)[sort_idx].numpy()
    y_hw = res_tensor.to(torch.float32)[sort_idx].numpy()

    allowed_intervals = _allowed_intervals_for(spec, x)
    # extra_undefined_ranges, when supplied (even as an empty list), fully
    # overrides the registry — letting callers inject custom asymptote bands or
    # suppress the registry's red shading entirely.
    if case.extra_undefined_ranges is not None:
        undefined_ranges = list(case.extra_undefined_ranges)
    else:
        undefined_ranges = list(
            _SFPU_UNDEFINED_RANGES.get(mathop, {}).get(Operand.A, [])
        )

    # ULP/eps spacing in _plot_and_print is taken from this format, so it must
    # match the format the compared values live in: golden and hw are produced
    # in output_format, so pass output_format (not input_format). Identical for
    # symmetric cases; for a mixed case like (Float32, Float16_b), using the
    # input format would measure ULP on the wrong (finer) grid and under-report.
    _plot_and_print(
        mathop,
        formats.output_format,
        x,
        y_golden,
        y_hw,
        plot_path,
        allowed_intervals=allowed_intervals,
        undefined_ranges=undefined_ranges,
    )

    test_passed = passed_test(golden_tensor, res_tensor, formats.output_format)
    logger.info("passed_test: {}", test_passed)

    # Bit-distance ULP measurement — reinterpret the fp32-promoted results as
    # int32 and take |golden_bits - hw_bits|; the max across finite samples is
    # the worst-case ULP error. Useful for sanity-checking accuracy claims.
    golden_fp32 = golden_tensor.to(torch.float32).contiguous().numpy()
    hw_fp32 = res_tensor.to(torch.float32).contiguous().numpy()
    finite_mask = np.isfinite(golden_fp32) & np.isfinite(hw_fp32)
    if finite_mask.any():
        gb = golden_fp32.view(np.int32)[finite_mask].astype(np.int64)
        rb = hw_fp32.view(np.int32)[finite_mask].astype(np.int64)
        int_min = np.iinfo(np.int32).min
        gb = np.where(gb < 0, int_min - gb, gb)
        rb = np.where(rb < 0, int_min - rb, rb)
        max_ulp = int(np.abs(gb - rb).max())
        logger.info(
            "[{}] max ULP across {} finite samples: {}",
            mathop.name,
            int(finite_mask.sum()),
            max_ulp,
        )
    else:
        logger.warning("[{}] no finite samples for ULP measurement", mathop.name)

    return test_passed


@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES, ids=[c.test_id for c in CASES])
def test_sfpu_stress(case: Case):
    passed = run_case(case)
    if case.expect_pass:
        assert passed, f"{case.test_id}: result did not match golden"
