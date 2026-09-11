# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive bfloat16 accuracy gate for the approximate tanh LUT.

Pushes every representable finite bf16 value -- all 65,279 of them, via
StimuliSpec.ulp_sweep -- through calculate_tanh<APPROXIMATION_MODE=true> in one
device run, and bounds the error against tanh computed in float64.

This is the only check that would notice the coefficient table regressing. The
ttnn tolerance test samples random inputs, and the unary sweep's per-format
tolerance is far looser than the table is accurate; neither pins the maximum.
The input set is fixed and the kernel is deterministic, so the bounds below are
tight rather than padded for sampling noise.

The reference is computed here rather than taken from UnarySFPUGolden: that
generator models the dest write and the pack back to L1, so it returns a value
already rounded to the output grid. Correct for a tolerance check, but it would
fold the pack's rounding into the ULP figure.
"""

from __future__ import annotations

import math

import numpy as np
import torch
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
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    get_num_blocks_and_num_tiles_in_block,
)
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
FORMATS = InputOutputFormat(BF16, BF16)
DEST_ACC = DestAccumulation.No

# Measured on a Wormhole n150: max 9.996 bf16 ULP, max abs error 0.0183516, both at
# fixed inputs. The headroom covers fp16 coefficient rounding and compiler drift,
# nothing more -- either 3-segment table this replaced fails both bounds by a wide
# margin (37.0 ULP / 0.1447 abs, and 48.0 / 0.0582).
MAX_ULP = 11.0
MAX_ABS = 0.021

# ULP is meaningless where the reference underflows the output format's resolution:
# ulp(tanh x) shrinks with x while the kernel error does not, so the ratio diverges
# near zero. Absolute error covers those points instead.
SCALE_FLOOR = 2.0**-8


def _sweep_dims(n_values: int) -> list[int]:
    """[rows, cols] holding every bf16 value, rounded up to whole dest blocks."""
    block_tiles = DEST_SYNC_TILE_LIMITS[DestSync.Half]
    tiles = max(1, math.ceil(n_values / (TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1])))
    if tiles > block_tiles:
        tiles = math.ceil(tiles / block_tiles) * block_tiles
    return [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tiles]


def test_tanh_approx_exhaustive_bf16():
    n_total = ulp_sweep_value_count(BF16, -math.inf, math.inf)
    dims = _sweep_dims(n_total)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=BF16,
        input_dimensions_A=dims,
        spec_A=StimuliSpec.ulp_sweep(low=-math.inf, high=math.inf),
        stimuli_format_B=BF16,
        input_dimensions_B=dims,
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DEST_ACC,
        FORMATS,
        dims,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        FORMATS,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.Yes),
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
            BF16,
            src_B,
            BF16,
            BF16,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DEST_ACC,
        unpack_to_dest=False,
    )
    result = torch.tensor(configuration.run().result, dtype=format_dict[BF16])

    # ulp_sweep zero-pads to fill the tensor; the padding is not test data.
    real = min(n_total, dims[0] * dims[1])
    order = torch.argsort(src_A.to(torch.float32)[:real])
    x = src_A.to(torch.float32)[:real][order].numpy().astype(np.float64)
    hw = result.to(torch.float32)[:real][order].numpy().astype(np.float64)
    ref = torch.tanh(torch.tensor(x, dtype=torch.float64)).numpy()

    assert x.size == n_total, f"swept {x.size} values, expected every one of {n_total}"
    assert np.isfinite(hw).all(), f"{int((~np.isfinite(hw)).sum())} non-finite outputs"

    abs_err = np.abs(hw - ref)
    mag = np.abs(ref)
    step = np.where(
        mag > 0, 2.0 ** (np.floor(np.log2(np.maximum(mag, 1e-300))) - 7), np.inf
    )
    ulp_err = np.where(mag >= SCALE_FLOOR, abs_err / step, 0.0)

    worst_ulp, worst_abs = int(np.argmax(ulp_err)), int(np.argmax(abs_err))
    logger.info(
        "tanh approx over {} bf16 values: max {:.3f} ULP at x={:.6g}, "
        "max |err| {:.7f} at x={:.6g}",
        x.size,
        ulp_err[worst_ulp],
        x[worst_ulp],
        abs_err[worst_abs],
        x[worst_abs],
    )

    drops = np.flatnonzero(np.diff(hw) < 0.0)
    assert drops.size == 0, (
        f"tanh must be nondecreasing; {drops.size} backward step(s), first from "
        f"x={x[drops[0]]:.6g} to {x[drops[0] + 1]:.6g} "
        f"({hw[drops[0]]:.7g} -> {hw[drops[0] + 1]:.7g})"
    )
    assert ulp_err[worst_ulp] <= MAX_ULP, (
        f"max {ulp_err[worst_ulp]:.3f} bf16 ULP exceeds {MAX_ULP} at x={x[worst_ulp]:.6g}: "
        f"tanh={ref[worst_ulp]:.9g}, hw={hw[worst_ulp]:.9g}"
    )
    assert abs_err[worst_abs] <= MAX_ABS, (
        f"max abs error {abs_err[worst_abs]:.7f} exceeds {MAX_ABS} at "
        f"x={x[worst_abs]:.6g}: tanh={ref[worst_abs]:.9g}, hw={hw[worst_abs]:.9g}"
    )
