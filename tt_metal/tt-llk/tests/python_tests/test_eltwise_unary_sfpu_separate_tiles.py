# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    DST_INDEX_IN,
    DST_INDEX_OUT,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

# Representative ops covering different write patterns:
# - SFPI dst_reg writes: abs, square, gelu, silu
# - TT_SFPSTORE writes (metal layer): handled via abs, square
SEPARATE_TILE_OPS = [
    MathOperation.Abs,
    MathOperation.Square,
    MathOperation.Gelu,
    MathOperation.Silu,
    MathOperation.Neg,
]

# Tile index pairs: (dst_index_in, dst_index_out)
# With SyncHalf and dest_acc=No, max tiles = 8 (indices 0-7)
# NOTE: dst_out must be >= dst_in because the offset (out - in) * 32 is computed
# as unsigned and used as a store immediate. Negative offsets underflow and are
# rejected by the compiler (out of sfpstore immediate range).
TILE_INDEX_PAIRS = [
    (0, 0),  # baseline: in-place (should match existing tests)
    (0, 1),  # write to next tile slot
    (0, 2),  # larger offset
    (1, 2),  # both non-zero, offset=1
]


@pytest.mark.parametrize("mathop", SEPARATE_TILE_OPS)
@pytest.mark.parametrize("dst_in,dst_out", TILE_INDEX_PAIRS)
def test_sfpu_separate_tiles(
    mathop: MathOperation,
    dst_in: int,
    dst_out: int,
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    input_dimensions = [32, 32]  # single tile
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No
    approx_mode = ApproximationMode.No

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
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

    num_blocks = tile_cnt_A
    num_tiles_in_block = 1  # process one tile per block for separate-tiles test

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_separate_tiles_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
            DST_INDEX_IN(dst_in),
            DST_INDEX_OUT(dst_out),
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
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Separate-tiles test failed for {mathop} with dst_in={dst_in}, dst_out={dst_out}"


# Test that exp in accurate mode (non-SFPLOADMACRO) works with separate tiles
@pytest.mark.parametrize("dst_in,dst_out", [(0, 1), (1, 2)])
def test_sfpu_separate_tiles_exp_accurate(dst_in: int, dst_out: int):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Exp,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks = tile_cnt_A
    num_tiles_in_block = 1

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_separate_tiles_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),  # accurate mode = no SFPLOADMACRO
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=MathOperation.Exp),
            DST_INDEX_IN(dst_in),
            DST_INDEX_OUT(dst_out),
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
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Exp accurate separate-tiles test failed with dst_in={dst_in}, dst_out={dst_out}"
