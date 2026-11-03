# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_coverage
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# The SDPA reduce supports only Float16_b and MAX column reduction.
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
DEST_ACC = DestAccumulation.No
MATHOP = MathOperation.ReduceColumn
REDUCE_POOL = ReducePool.Max
INPUT_DIMENSIONS = [512, 64]  # four independent 4x2 subblocks

# all_negative catches a max seed that is not really -inf.
STIMULI = {
    "default": None,
    "all_negative": StimuliSpec.uniform(low=-2.0, high=-0.5),
}


# Has a compilation error on coverage, https://github.com/tenstorrent/tt-llk/issues/884
@skip_for_coverage
# mathop stays a parameter so `--op ReduceColumn` still selects these cases.
@parametrize(stimuli=list(STIMULI), mathop=[MATHOP])
def test_sfpu_reduce_sdpa(stimuli, mathop):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=FORMATS.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=FORMATS.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
        spec_A=STIMULI[stimuli],
    )

    src_A = tilize_block(src_A, INPUT_DIMENSIONS).flatten()

    # GOLDEN GENERATION
    # *******************************************************

    # Undo tilization so src_A is standard [32, 32]
    src_A_untilized = untilize_block(src_A, FORMATS.input_format, INPUT_DIMENSIONS)

    # Each destination block is an independent 4x2 SDPA reduction subblock.
    golden_tensor = torch.zeros_like(src_A_untilized)
    subblock_rows = 4 * 32
    for row in range(0, INPUT_DIMENSIONS[0], subblock_rows):
        golden_tensor[row, :] = torch.max(
            src_A_untilized[row : row + subblock_rows, :], dim=0
        ).values

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, DEST_ACC, FORMATS, INPUT_DIMENSIONS, TILE_DIMENSIONS
    )

    # *******************************************************

    configuration = TestConfig(
        "sources/sfpu_reduce_sdpa_test.cpp",
        FORMATS,
        templates=[
            generate_input_dim(
                INPUT_DIMENSIONS, INPUT_DIMENSIONS, block_ct_dim=2, block_rt_dim=4
            ),
            MATH_OP(mathop=mathop, pool_type=REDUCE_POOL),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=False,  # Must be False since math kernel does A2D copy
        dest_acc=DEST_ACC,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[FORMATS.output_format])
    res_tensor = untilize_block(res_tensor, FORMATS.output_format, INPUT_DIMENSIONS)

    for row in range(0, INPUT_DIMENSIONS[0], subblock_rows):
        assert passed_test(golden_tensor[row], res_tensor[row], FORMATS.output_format)
