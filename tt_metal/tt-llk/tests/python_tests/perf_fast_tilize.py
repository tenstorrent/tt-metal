# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)


def _fast_tilize_tile_dims(dest_acc):
    """Dest-full tall/wide in tile counts, plus ct=2 for the narrow fast path."""
    tiles = [
        [dims[0] // 32, dims[1] // 32]
        for dims in generate_perf_input_dimensions(dest_acc, DestSync.Half)
    ]
    if [1, 2] not in tiles:
        tiles.append([1, 2])
    return tiles


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    input_format=[DataFormat.Float32, DataFormat.Float16_b],
    output_format=[DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b],
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    input_dimensions=lambda dest_acc: _fast_tilize_tile_dims(dest_acc),
)
def test_fast_tilize_perf(
    perf_report,
    input_format,
    output_format,
    dest_acc,
    input_dimensions,
):
    tile_count = input_dimensions[0] * input_dimensions[1]
    input_dimensions = (input_dimensions[0] * 32, input_dimensions[1] * 32)

    formats = InputOutputFormat(input_format, output_format)

    configuration = PerfConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(1024),
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
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    configuration.run(perf_report, run_count=2)
