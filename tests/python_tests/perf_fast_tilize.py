# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole
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


def generate_input_dimensions(max_size: int) -> list[tuple[int, int]]:
    """
    Generates a list of tuples representing width and height in tiles for input tensors,
    up to the specified maximum size in tiles.
    For tilize tensor width is important so all widths from 1 to max_size are generated.
    In the interest of reducing the number of test cases, instead of generating all possible heights
    critical subsets of valid heights are generated (three smallest, three largest and three middle heights).
    Parameters:
    max_size (int): Maximum number of tiles the resulting tensor can have.
    Returns:
    List[tuple[int, int]]: A list of tuples representing the width and height of the input tensor in tiles.
    """
    dimensions = []
    for width in range(1, max_size + 1):
        max_height = max_size // width
        heights = [
            1,
            2,
            3,
            (max_height // 2) - 1,
            max_height // 2,
            (max_height // 2) + 1,
            max_height - 2,
            max_height - 1,
            max_height,
        ]
        heights = [h for h in heights if h > 0 and h <= max_height]
        heights = list(set(heights))
        for height in heights:
            dimensions.append((width, height))
    return dimensions


@skip_for_blackhole
@pytest.mark.perf
@parametrize(
    input_format=[DataFormat.Float32, DataFormat.Float16_b],
    output_format=[DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b],
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    input_dimensions=generate_input_dimensions(16),
)
def test_fast_tilize_perf(
    perf_report,
    input_format,
    output_format,
    dest_acc,
    input_dimensions,
    workers_tensix_coordinates,
):
    tile_count = input_dimensions[0] * input_dimensions[1]
    input_dimensions = (input_dimensions[0] * 32, input_dimensions[1] * 32)

    formats = InputOutputFormat(input_format, output_format)

    configuration = PerfConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[generate_input_dim(input_dimensions, input_dimensions)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(1024), NUM_FACES(4)],
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
    )

    configuration.run(perf_report, run_count=2, location=workers_tensix_coordinates)
