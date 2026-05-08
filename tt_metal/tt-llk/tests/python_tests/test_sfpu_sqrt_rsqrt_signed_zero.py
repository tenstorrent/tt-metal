# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
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

SPECIAL_VALUES = torch.tensor(
    [-0.0, 0.0, -1.0, -float("inf"), float("inf")],
    dtype=torch.float32,
)


def _hex_bits(tensor):
    return [f"0x{int(v) & 0xffffffff:08x}" for v in tensor.view(torch.int32)]


@pytest.mark.parametrize("mathop", [MathOperation.Sqrt, MathOperation.Rsqrt])
@pytest.mark.parametrize("approx_mode", [ApproximationMode.No, ApproximationMode.Yes])
def test_sqrt_rsqrt_signed_zero_specials_float32(mathop, approx_mode):
    input_dimensions = [32, 32]
    num_elements = input_dimensions[0] * input_dimensions[1]
    src_a = SPECIAL_VALUES.repeat(num_elements // SPECIAL_VALUES.numel() + 1)[
        :num_elements
    ]
    src_b = torch.zeros_like(src_a)
    tile_count = 1
    sample_count = SPECIAL_VALUES.numel()

    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    expected = torch.sqrt(src_a) if mathop == MathOperation.Sqrt else torch.rsqrt(src_a)

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
            APPROX_MODE(approx_mode),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_a,
            formats.input_format,
            src_b,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )

    actual = torch.tensor(configuration.run().result, dtype=torch.float32)

    actual_bits = actual.view(torch.int32)
    expected_bits = expected.view(torch.int32)
    same_bits = actual_bits == expected_bits
    both_nan = torch.isnan(actual) & torch.isnan(expected)
    same = same_bits | both_nan

    assert torch.all(same), (
        f"{mathop.name} {approx_mode.name} special-value mismatch\n"
        f"input first {sample_count}:    {SPECIAL_VALUES.tolist()}\n"
        f"expected first {sample_count}: {expected[:sample_count].tolist()} "
        f"{_hex_bits(expected[:sample_count])}\n"
        f"actual first {sample_count}:   {actual[:sample_count].tolist()} "
        f"{_hex_bits(actual[:sample_count])}\n"
        f"bad indices:      {torch.nonzero(~same).flatten()[:32].tolist()}"
    )
