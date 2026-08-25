# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import (
    generate_perf_input_dimensions,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IN_TILE_DIMS,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    generate_input_dim,
)


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    dest_sync=[DestSync.Half],
    r_dim=[16],
    input_dimensions=lambda dest_acc, dest_sync: generate_perf_input_dimensions(
        dest_acc, dest_sync
    ),
)
def test_perf_pack_untilize(
    perf_report, formats, dest_acc, dest_sync, r_dim, input_dimensions
):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync, dest_acc, formats, input_dimensions
    )

    configuration = PerfConfig(
        "sources/dense_pack_untilize_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            NUM_FACES(4),
            IN_TILE_DIMS(r_dim, 32, 32, 32),
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
    configuration.run(perf_report)
