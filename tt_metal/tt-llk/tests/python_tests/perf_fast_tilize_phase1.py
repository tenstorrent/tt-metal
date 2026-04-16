# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 performance test for BH fast-tilize unpack.
Measures cycles per tile for the unpack+math+pack pipeline.
Target: ≤ 20 cycles/tile (amortized across 4-tile unit).
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)


@pytest.mark.perf
@parametrize(
    input_format=[DataFormat.Float16_b],
    output_format=[DataFormat.Float16_b],
    dest_acc=[DestAccumulation.No],
    dimensions=[(1, 4), (1, 8), (1, 16), (2, 8), (4, 4)],
)
def test_fast_tilize_unpack_perf(
    perf_report,
    input_format,
    output_format,
    dest_acc,
    dimensions,
    workers_tensix_coordinates,
):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("BH only")

    input_height_tiles, input_width_tiles = dimensions
    assert input_width_tiles % 4 == 0, "ct_dim must be divisible by 4"

    tile_count = input_height_tiles * input_width_tiles
    # Output tiles: 8 per unit of 4 input tiles
    num_output_tiles = input_height_tiles * (input_width_tiles // 4) * 8
    input_dims = (input_height_tiles * 32, input_width_tiles * 32)

    formats = InputOutputFormat(input_format, output_format)

    configuration = PerfConfig(
        "sources/fast_tilize_phase1_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=[
            generate_input_dim(input_dims, input_dims),
            TILE_COUNT(num_output_tiles),
            LOOP_FACTOR(4),
            NUM_FACES(4),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=num_output_tiles,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report, run_count=2, location=workers_tensix_coordinates)
