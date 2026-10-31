# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The row pack (_llk_pack_rows_) of one to 64 rows per tile, four tiles per half-sync DEST section (pack_rows_perf_test.cpp)."""

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import FACES_PER_TILE, TILE_DIMENSIONS
from helpers.llk_params import BlocksCalculationAlgorithm, DestAccumulation, DestSync, PackerReluType
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_ROWS_TO_PACK,
    NUM_TILES_IN_BLOCK,
    PACK_SRC_FROM_OUTPUT,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)

F = DataFormat
IO = InputOutputFormat

ROWS_FORMATS = [IO(F.Float16_b, F.Float16_b), IO(F.Float16, F.Float16), IO(F.Float32, F.Float32), IO(F.Int32, F.Int32), IO(F.Bfp8_b, F.Bfp8_b), IO(F.UInt16, F.UInt16)]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=ROWS_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    input_dimensions=[[64, 64]],
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half],
    num_rows_to_pack=[1, 2, 3, 4, 5, 8, 16, 17, 32, 50, 51, 64],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_grid_rows(perf_report, formats, dest_acc, input_dimensions, relu_type, dest_sync, num_rows_to_pack, run_types, loop_factor, is_perf):
    """_llk_pack_rows_ of num_rows_to_pack rows per tile, four tiles per DEST section."""
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    unpack_to_dest = formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync, dest_acc, formats, input_dimensions, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    configuration = create_test_or_perf_config(
        is_perf=True,
        run_types=run_types,
        test_config_kwargs={
            "test_name": "sources/pack_rows_perf_test.cpp",
            "formats": formats,
            "templates": [generate_input_dim(input_dimensions, input_dimensions), TILIZE(), DEST_SYNC(dest_sync), PACK_SRC_FROM_OUTPUT()],
            "runtimes": [
                TILE_COUNT(tile_cnt_A),
                DEST_INDEX(0),
                RELU_CONFIG(0),
                NUM_FACES(num_faces=FACES_PER_TILE),
                NUM_BLOCKS(num_blocks),
                NUM_TILES_IN_BLOCK(num_tiles_in_block),
                NUM_ROWS_TO_PACK(num_rows_to_pack),
                LOOP_FACTOR(loop_factor),
            ],
            "variant_stimuli": StimuliConfig(
                src_A, formats.input_format, src_B, formats.input_format, formats.output_format, tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": unpack_to_dest,
        },
    )
    configuration.run(perf_report)
