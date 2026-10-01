# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    PerfRunType,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
)

# A 32-bit input with dest accumulation selects the unpack-to-dest broadcast path.
PERF_FORMATS = input_output_formats([DataFormat.Float32, DataFormat.Int32], same=True)

# Eight 32-bit tiles fill both DEST halves in two blocks of four, so every DEST tile slot is measured.
NUM_TILES = 8


@pytest.mark.perf
@parametrize(
    formats=PERF_FORMATS,
    broadcast_type=[
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    loop_factor=[32],
)
def test_perf_unpack_bcast(perf_report, formats, broadcast_type, loop_factor):
    input_dimensions = [TILE_DIMENSIONS[0] * NUM_TILES, TILE_DIMENSIONS[1]]
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.Yes,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = PerfConfig(
        "sources/unpack_bcast_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[BROADCAST_TYPE(broadcast_type)],
        runtimes=[
            NUM_FACES(4),
            TILE_COUNT(NUM_TILES),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=NUM_TILES,
            tile_count_B=NUM_TILES,
            tile_count_res=NUM_TILES,
            num_faces=4,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run(perf_report)
