# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Legacy LUT tanh derivative (`tanh_derivative_lut`) on non-finite inputs.

`_calculate_tanh_derivative_` computes 1 - lut(x)^2, and the LUT's last segment is
the constant pair (A=0, B=1), evaluated as A*|x| + B. An infinite input used to
compute 0 * inf + 1 = NaN, so tanh'(+-inf) came out NaN instead of 0
(tenstorrent/tt-llk#1701 item 10). The finite stimulus range of the main sweep is
[-3, 3], so nothing else reaches the tail with an infinity.

Blackhole now steps +-inf down to +-FLT_MAX before the LUT. Wormhole keeps the old
behaviour, so this module is Blackhole-only.

Inputs are written as raw bit patterns through an integer view, so both NaN signs
and the extreme payloads reach L1 as asked. Float32 in, Float32 out with a 32-bit
Dest is the pipeline that delivers a NaN to the SFPU intact (a bfloat16 NaN
unpacked into a 32-bit Dest arrives as an infinity), so that arm also pins NaN
propagation.

Reference: tanh'(+-inf) = 0, tanh'(NaN) = NaN, tanh'(+-0) = 1.
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

pytestmark = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.BLACKHOLE,
    reason="Wormhole still returns NaN for +-inf here (tenstorrent/tt-llk#1701 item 10)",
)

# key -> (fp32 bits, bfloat16 bits)
SPECIALS = {
    "+inf": (0x7F800000, 0x7F80),
    "-inf": (0xFF800000, 0xFF80),
    "+NaN": (0x7FC00000, 0x7FC0),
    "-NaN": (0xFFC00000, 0xFFC0),
    "+NaN(min payload)": (0x7F800001, 0x7F81),
    "-NaN(max payload)": (0xFFFFFFFF, 0xFFFF),
    "+max": (0x7F7FFFFF, 0x7F7F),
    "-max": (0xFF7FFFFF, 0xFF7F),
    "+0.0": (0x00000000, 0x0000),
    "-0.0": (0x80000000, 0x8000),
}

EXPECTED = {
    DataFormat.Float32: {
        "+inf": "+0.0",
        "-inf": "+0.0",
        "+NaN": "NaN",
        "-NaN": "NaN",
        "+NaN(min payload)": "NaN",
        "-NaN(max payload)": "NaN",
        "+max": "+0.0",
        "-max": "+0.0",
        "+0.0": "1",
        "-0.0": "1",
    },
    # A NaN does not survive the bfloat16 path to L1, so that arm pins the
    # infinities and the finite saturation only.
    DataFormat.Float16_b: {
        "+inf": "+0.0",
        "-inf": "+0.0",
        "+max": "+0.0",
        "-max": "+0.0",
        "+0.0": "1",
        "-0.0": "1",
    },
}


def _run(fmt, dest_acc):
    formats = InputOutputFormat(fmt, fmt)
    dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tc_A, src_B, tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=StimuliSpec.constant(0.5),
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    flat = src_A.view(-1)
    if fmt == DataFormat.Float32:
        bits = flat.view(torch.int32)
        patterns = [
            int(np.uint32(fp32).view(np.int32)) for fp32, _ in SPECIALS.values()
        ]
        bits[: len(patterns)] = torch.tensor(patterns, dtype=torch.int32)
    else:
        bits = flat.view(torch.uint16)
        patterns = [bf16 for _, bf16 in SPECIALS.values()]
        bits[: len(patterns)] = torch.tensor(patterns, dtype=torch.uint16)

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
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.TanhDerivativeLut),
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
        unpack_to_dest=fmt.is_32_bit() and dest_acc == DestAccumulation.Yes,
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
    "fmt,dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16", "fp32"],
)
def test_tanh_derivative_lut_specials(fmt, dest_acc):
    hw = _run(fmt, dest_acc)
    got = {key: _classify(hw[i]) for i, key in enumerate(SPECIALS)}

    # The rest of the tile is 0.5, which every pipeline returns as a finite value
    # in (0, 1); a tile that came back all zero or all NaN never ran the kernel.
    rest = hw[len(SPECIALS) :]
    assert np.isfinite(rest).all() and (rest > 0.5).all() and (rest < 1.0).all(), (
        "the 0.5 background did not come back as tanh'(0.5) ~ 0.79: %s"
        % np.unique(rest)[:8]
    )

    wrong = {k: (got[k], want) for k, want in EXPECTED[fmt].items() if got[k] != want}
    assert not wrong, (
        "tanh_derivative_lut on special inputs (got, expected): %s" % wrong
    )
