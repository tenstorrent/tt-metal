# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive Float16_b ULP sweep of the metal hw/ckernels tanh kernels.

Feeds every representable finite bf16 value (65,279 of them, deduplicated and
sorted by StimuliSpec.ulp_sweep) through calculate_tanh on silicon and reports
the error against the torch/libm golden. Covers all three kernel paths:

  approx-lut   APPROXIMATION_MODE=true            -> the 3-segment SFPLUT
  poly         approx=false, dest_acc=No          -> _sfpu_tanh_polynomial_x2_
  fp32-exact   approx=false, dest_acc=Yes         -> _sfpu_tanh_fp32_accurate_

Run:  CHIP_ARCH=wormhole .venv/bin/python -m pytest test_tanh_ulp_sweep.py -s

Measures; it does not gate. Set $TANH_SWEEP_OUT to also write one JSON summary
per config, tagged with $TANH_SWEEP_TAG, so a before/after pair of coefficient
tables can be swept into one directory and diffed offline.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from helpers.accuracy_metrics import compute_pointwise_metrics
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.stimuli_generator.strategies.structured import ulp_sweep_value_count
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)

BF16 = DataFormat.Float16_b
FP32 = DataFormat.Float32

_TILE_ELEMENTS = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]

# Scale-relative metrics (ULP, relative error) are meaningless where the golden
# is tiny: tanh(x) ~ x there, so the local ULP collapses with the value while the
# kernel's error does not. Gate them at 2^-8, the bf16 mantissa step.
_SCALE_FLOOR = 2.0**-8

# Set TANH_SWEEP_OUT to keep a machine-readable summary per config; unset, the
# test only logs. TANH_SWEEP_TAG names the run, so a before/after pair of tables
# can be swept into one directory and diffed offline.
_OUT_ENV = os.getenv("TANH_SWEEP_OUT")
OUT_DIR = Path(_OUT_ENV) if _OUT_ENV else None
TAG = os.getenv("TANH_SWEEP_TAG", "run")


@dataclass(frozen=True)
class Config:
    """One kernel path to sweep."""

    label: str
    out_fmt: DataFormat
    approx: ApproximationMode
    dest_acc: DestAccumulation

    @property
    def formats(self) -> InputOutputFormat:
        return InputOutputFormat(BF16, self.out_fmt)

    @property
    def slug(self) -> str:
        return self.label.replace(" ", "-")


CONFIGS: List[Config] = [
    # The approximate LUT path, as a consumer sees it (bf16 in, bf16 out).
    Config("approx-lut-bf16-out", BF16, ApproximationMode.Yes, DestAccumulation.No),
    # The same LUT read out of an fp32 dest, so the output format's own rounding
    # does not mask part of the kernel error.
    Config("approx-lut-fp32-out", FP32, ApproximationMode.Yes, DestAccumulation.Yes),
    # Reference points: the two accurate paths, for scale.
    Config("poly-bf16-out", BF16, ApproximationMode.No, DestAccumulation.No),
    Config("fp32-exact-fp32-out", FP32, ApproximationMode.No, DestAccumulation.Yes),
]


def _sweep_dims(dest_acc: DestAccumulation, n_values: int) -> List[int]:
    """[rows, cols] holding every bf16 value, rounded up to whole dest blocks.

    Mirrors test_sfpu_plot._ulp_sweep_dims: a bf16 sweep of 65,279 values needs
    64 tiles, which spans several dest blocks, and the block count has to divide
    the tile count evenly.
    """
    from helpers.param_config import DEST_SYNC_TILE_LIMITS

    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    block_tiles = DEST_SYNC_TILE_LIMITS[DestSync.Half] // capacity_divisor

    tiles = max(1, math.ceil(n_values / _TILE_ELEMENTS))
    if tiles > block_tiles:
        tiles = math.ceil(tiles / block_tiles) * block_tiles
    return [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tiles]


def _run_on_device(cfg: Config, dims: List[int]):
    """Build the exhaustive sweep, run it, and return (input, hw result)."""
    formats = cfg.formats
    spec = StimuliSpec.ulp_sweep(low=-math.inf, high=math.inf)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=spec,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        cfg.dest_acc,
        formats,
        dims,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(cfg.approx),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.Tanh),
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
        dest_acc=cfg.dest_acc,
        # Only a 32-bit *input* is unpacked straight to dest; bf16 goes through srcA.
        unpack_to_dest=False,
    )
    hw = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    return src_A, hw


def _gaussian_weights(x: np.ndarray) -> np.ndarray:
    """Midpoint-quadrature N(0,1) weight per swept point.

    A uniform mean over the bf16 grid is not a fair accuracy summary: half the
    grid sits at |x| >= 2, where tanh has already saturated and every kernel is
    bit-exact. Weighting each point by phi(x) times the width of its bf16 cell
    reports the error an activation at that distribution would actually see.
    """
    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0]
    edges[-1] = x[-1]
    width = np.diff(edges)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    w = phi * width
    total = w.sum()
    return w / total if total > 0 else w


def torch_reference(x: np.ndarray) -> np.ndarray:
    """The torch golden: tanh of the exact bf16 input, evaluated in float64.

    Deliberately *not* UnarySFPUGolden: that models the dest write and the pack
    back to L1, so it returns a value already rounded to the dest/output grid.
    That is the right reference for a pass/fail tolerance check, but it corrupts
    an ULP measurement -- on the fp32-output configs it made the fp32-accurate
    path read as 3e4 ULP out when it is really within ~1 ULP, because the
    "error" being measured was the golden's own bf16 rounding.
    """
    return torch.tanh(torch.tensor(x, dtype=torch.float64)).numpy()


def _summarize(cfg: Config, x: np.ndarray, ref: np.ndarray, hw: np.ndarray) -> dict:
    """Error of *hw* against the float64 torch reference *ref*.

    ULP is measured on the output format's grid at the true value, so a
    correctly-rounded kernel scores <= 0.5 ULP.
    """
    m = compute_pointwise_metrics(x, ref, hw, cfg.out_fmt)
    signed_err = m["signed_error"]
    abs_err = np.abs(signed_err)
    ulp = np.abs(m["signed_ulp_error"])

    # What the output format can actually represent: the reference rounded to it.
    ref_out = (
        torch.tensor(ref, dtype=torch.float64)
        .to(format_dict[cfg.out_fmt])
        .to(torch.float64)
        .numpy()
    )

    finite = m["is_finite_hw"] & m["is_finite_golden"]
    # ULP/relative error only where the reference is large enough to have scale.
    gated = finite & (np.abs(ref) >= _SCALE_FLOOR) & np.isfinite(ulp)

    worst = int(np.argmax(np.where(finite, abs_err, -1.0)))
    worst_ulp = int(np.argmax(np.where(gated, ulp, -1.0)))
    rel = m["rel_error"]
    worst_rel = int(np.argmax(np.where(gated & np.isfinite(rel), rel, -1.0)))

    w = _gaussian_weights(x)
    wrms = float(math.sqrt(np.sum(w * np.where(finite, abs_err, 0.0) ** 2)))

    # tanh is nondecreasing; the sweep is sorted, so a drop is a real inversion.
    drops = np.flatnonzero(np.diff(hw[finite]) < 0.0)

    # Top offenders, so a headline number can be traced back to real points.
    order = np.argsort(np.where(finite, abs_err, -1.0))[::-1][:10]
    worst_points = [
        {
            "x": float(x[i]),
            "ref": float(ref[i]),
            "ref_rounded": float(ref_out[i]),
            "hw": float(hw[i]),
            "abs_err": float(abs_err[i]),
            "ulp": float(ulp[i]),
        }
        for i in order
    ]

    g = ulp[gated]
    return {
        "worst_points": worst_points,
        "tag": TAG,
        "label": cfg.label,
        "arch": str(TestConfig.CHIP_ARCH.name).lower(),
        "in_fmt": "Float16_b",
        "out_fmt": cfg.out_fmt.name,
        "approx": cfg.approx == ApproximationMode.Yes,
        "dest_acc": cfg.dest_acc == DestAccumulation.Yes,
        "n_values": int(x.size),
        "n_scale_gated": int(gated.sum()),
        "correctly_rounded_frac": float(np.mean(hw[finite] == ref_out[finite])),
        "max_abs_err": float(abs_err[finite].max()),
        "max_abs_err_at_x": float(x[worst]),
        "gauss_weighted_rms_abs_err": wrms,
        "max_ulp": float(g.max()) if g.size else float("nan"),
        "max_ulp_at_x": float(x[worst_ulp]),
        "p999_ulp": float(np.percentile(g, 99.9)) if g.size else float("nan"),
        "p99_ulp": float(np.percentile(g, 99.0)) if g.size else float("nan"),
        "median_ulp": float(np.median(g)) if g.size else float("nan"),
        "mean_ulp": float(g.mean()) if g.size else float("nan"),
        "max_rel_err": float(np.nanmax(rel[gated])) if g.size else float("nan"),
        "max_rel_err_at_x": float(x[worst_rel]),
        "monotonicity_violations": int(drops.size),
        # Absolute size of the largest backwards step, to tell a real inversion
        # from a sub-ULP wiggle in an otherwise correctly-rounded kernel.
        "monotonicity_worst_drop": (
            float(np.max(-np.diff(hw[finite])[drops])) if drops.size else 0.0
        ),
        "n_nonfinite_hw": int((~m["is_finite_hw"]).sum()),
    }


def _print_summary(s: dict) -> None:
    logger.info("── {} ({} -> {}) ──", s["label"], s["in_fmt"], s["out_fmt"])
    logger.info(
        "   swept {} values, {} above the 2^-8 scale floor",
        s["n_values"],
        s["n_scale_gated"],
    )
    logger.info("   correctly rounded          {:.2%}", s["correctly_rounded_frac"])
    logger.info(
        "   max |abs err|              {:.6g}  (at x = {:.6g})",
        s["max_abs_err"],
        s["max_abs_err_at_x"],
    )
    logger.info("   N(0,1)-weighted RMS err    {:.6g}", s["gauss_weighted_rms_abs_err"])
    logger.info(
        "   ULP err  max {:.6g}  p99.9 {:.6g}  p99 {:.6g}  median {:.6g}  mean {:.6g}",
        s["max_ulp"],
        s["p999_ulp"],
        s["p99_ulp"],
        s["median_ulp"],
        s["mean_ulp"],
    )
    logger.info(
        "   max rel err                {:.6g}  (at x = {:.6g})",
        s["max_rel_err"],
        s["max_rel_err_at_x"],
    )
    logger.info("   monotonicity violations    {}", s["monotonicity_violations"])


@pytest.mark.accuracy
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.slug)
def test_tanh_ulp_sweep(cfg: Config):
    n_total = ulp_sweep_value_count(BF16, -math.inf, math.inf)
    dims = _sweep_dims(cfg.dest_acc, n_total)

    src_A, hw_t = _run_on_device(cfg, dims)

    # ulp_sweep zero-pads to fill the tensor; padding is not test data.
    real = min(n_total, dims[0] * dims[1])
    order = torch.argsort(src_A.to(torch.float32)[:real])
    x = src_A.to(torch.float32)[:real][order].numpy().astype(np.float64)
    hw = hw_t.to(torch.float32)[:real][order].numpy().astype(np.float64)

    ref = torch_reference(x)
    s = _summarize(cfg, x, ref, hw)
    _print_summary(s)

    # Full per-point arrays, for plotting the curve rather than summarising it.
    if os.getenv("TANH_SWEEP_DUMP"):
        d = Path(os.getenv("TANH_SWEEP_DUMP"))
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{TAG}__{cfg.slug}.npz", x=x, ref=ref, hw=hw)

    if OUT_DIR is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{TAG}__{cfg.slug}.json"
        path.write_text(json.dumps(s, indent=2) + "\n")
        logger.info("   wrote {}", path)

    # Sanity only — this file measures, it does not gate. A dead kernel or a
    # sweep that never reached the device would show up here.
    assert s["n_values"] == n_total, "sweep did not cover every bf16 value"
    assert (
        s["correctly_rounded_frac"] > 0.1
    ), "kernel output looks unrelated to the reference"
