# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM, TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PerfRunType,
    ReducePool,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block
from test_sfpu_reduce_multidim import use_int32_twos_complement


@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
    ],
    dest_acc=[DestAccumulation.Yes],
    dest_sync=[DestSync.Half],
    reduce_pool=[ReducePool.Max, ReducePool.Sum],
    num_row_tiles=[4],
)
def test_perf_sfpu_reduce_multidim(
    perf_report, formats, dest_acc, dest_sync, reduce_pool, num_row_tiles
):
    input_dimensions = [num_row_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = num_row_tiles
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    if num_blocks != 1:
        pytest.skip("Row reduction requires all tiles in one dest section")

    torch_format = format_dict[formats.input_format]
    stimuli_size = tile_cnt * ELEMENTS_PER_TILE
    if formats.input_format.is_integer():
        src_A = torch.randint(-32, 33, (stimuli_size,), dtype=torch_format)
    else:
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
    src_B = torch.zeros_like(src_A)
    src_A = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = PerfConfig(
        "sources/sfpu_reduce_multidim_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            REDUCE_POOL_TYPE(reduce_pool),
        ],
        runtimes=[
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            TILE_COUNT(tile_cnt),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
            twos_complement=use_int32_twos_complement(formats),
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
