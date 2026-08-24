# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestSync,
    PerfRunType,
)
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
    NUM_BLOCKS,
    NUM_ROWS_TO_PACK,
    NUM_TILES_IN_BLOCK,
    generate_input_dim,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    num_rows_to_pack=[1, 16, 64],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(
        dest_acc, DestSync.Half
    ),
)
def test_perf_pack_rows(
    perf_report, formats, dest_acc, num_rows_to_pack, input_dimensions
):
    try:
        num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
            DestSync.Half,
            dest_acc,
            formats,
            input_dimensions,
            [32, 32],
            BlocksCalculationAlgorithm.Standard,
        )
    except ValueError as e:
        pytest.skip(f"Skipping incompatible dimension: {e}")

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = PerfConfig(
        "sources/pack_rows_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[generate_input_dim(input_dimensions, input_dimensions)],
        runtimes=[
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_ROWS_TO_PACK(num_rows_to_pack),
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
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    configuration.run(perf_report)
