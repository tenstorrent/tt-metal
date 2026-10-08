# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GELU derivative on the bit patterns a finite-value sweep never presents.

`calculate_gelu_derivative_polynomial` picks its region with a chain of ordered
float compares (x >= 3.1719 -> 1.0, x >= -3 -> polynomial, x > -13.375 -> tail,
else 0). SFPU compares order NaNs by sign and magnitude, so +NaN sorts above +inf
and -NaN below -inf: without an explicit case +NaN comes out as 1.0 and -NaN as 0
instead of propagating (tenstorrent/tt-llk#1701 item 12).

Blackhole propagates NaN with an fp32 destination. The bfloat16-destination arm
keeps 1.0 / 0 on both architectures: it ends in convert<vFloat16b>
(SFPSTOCHRND), which turns any NaN into an infinity, so it could not return NaN
anyway. Wormhole keeps the old behaviour on both arms.

Inputs are written as raw bit patterns through an integer view, so both NaN signs
and the two extreme payloads (0x7F800001 / 0xFFFFFFFF for fp32, 0x7F81 / 0xFFFF for
bfloat16) reach L1 as asked -- torch's float -> bfloat16 conversion would collapse
every NaN onto one positive pattern.

Reference: GELU'(+inf) = 1, GELU'(-inf) = 0, GELU'(+-0) = 0.5, GELU'(NaN) = NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
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
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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

# key -> (fp32 bits, bfloat16 bits)
SPECIALS = {
    "+inf": (0x7F800000, 0x7F80),
    "-inf": (0xFF800000, 0xFF80),
    "+NaN": (0x7FC00000, 0x7FC0),
    "-NaN": (0xFFC00000, 0xFFC0),
    "+NaN(min payload)": (0x7F800001, 0x7F81),
    "-NaN(max payload)": (0xFFFFFFFF, 0xFFFF),
    "+0.0": (0x00000000, 0x0000),
    "-0.0": (0x80000000, 0x8000),
}

# Finite controls on either side of each region boundary; their outputs are only
# checked for being finite, which pins that the NaN case did not leak into them.
CONTROLS = [3.1719, 3.0, 5.0, 1.0e30, -3.0, -3.25, -13.0, -13.375, -20.0, -1.0e30]


def _run(out_fmt, dest_acc, approx):
    in_fmt = out_fmt
    formats = InputOutputFormat(in_fmt, out_fmt)
    dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tc_A, src_B, tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=StimuliSpec.constant(0.0),
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    flat = src_A.view(-1)
    if in_fmt == DataFormat.Float32:
        bits = flat.view(torch.int32)
        patterns = [
            np.int32(np.uint32(fp32).view(np.int32)) for fp32, _ in SPECIALS.values()
        ]
        bits[: len(patterns)] = torch.tensor(patterns, dtype=torch.int32)
    else:
        bits = flat.view(torch.uint16)
        patterns = [bf16 for _, bf16 in SPECIALS.values()]
        bits[: len(patterns)] = torch.tensor(patterns, dtype=torch.uint16)
    base = len(SPECIALS)
    flat[base : base + len(CONTROLS)] = torch.tensor(CONTROLS, dtype=flat.dtype)

    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        dims,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(approx),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.GeluDerivative),
        ],
        runtimes=[TILE_COUNT(tc_A), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tc_A,
            tile_count_B=tc_B,
            tile_count_res=tc_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=in_fmt.is_32_bit() and dest_acc == DestAccumulation.Yes,
    )
    hw = torch.tensor(cfg.run().result, dtype=format_dict[formats.output_format])
    return hw.to(torch.float32).numpy()


def _classify(v):
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
        return "+inf" if v > 0 else "-inf"
    if v == 0.0:
        return "-0.0" if np.signbit(v) else "+0.0"
    return "%.9g" % v


@pytest.mark.parametrize(
    "approx", [ApproximationMode.No, ApproximationMode.Yes], ids=["exact", "approx"]
)
@pytest.mark.parametrize(
    "out_fmt,dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16", "fp32"],
)
def test_gelu_derivative_specials(out_fmt, dest_acc, approx):
    hw = _run(out_fmt, dest_acc, approx)
    got = {key: _classify(hw[i]) for i, key in enumerate(SPECIALS)}

    controls = hw[len(SPECIALS) : len(SPECIALS) + len(CONTROLS)]
    assert np.isfinite(controls).all(), "finite controls %s gave %s" % (
        CONTROLS,
        controls,
    )
    # The probes must have arrived: the -0.0/+0.0 lanes evaluate the polynomial
    # to 0.5, and a stimuli path that lost them would leave the whole tile at 0.
    assert got["+0.0"] == "0.5", "the +0.0 probe did not reach the kernel: %s" % got

    propagates = (
        get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and out_fmt == DataFormat.Float32
    )
    nan = "NaN" if propagates else None
    expected = {
        "+inf": "1",
        "-inf": "+0.0",
        "+NaN": nan or "1",
        "-NaN": nan or "+0.0",
        "+NaN(min payload)": nan or "1",
        "-NaN(max payload)": nan or "+0.0",
        "+0.0": "0.5",
        "-0.0": "0.5",
    }

    wrong = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
    assert not wrong, "GELU' on special inputs (got, expected): %s" % wrong
