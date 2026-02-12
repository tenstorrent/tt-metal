# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.golden_generators import TilizeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.utils import passed_test


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
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b]
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    dimensions=generate_input_dimensions(25),
)
def test_fast_tilize(formats, dest_acc, dimensions, workers_tensix_coordinates):

    input_width, input_height = dimensions

    if formats.input == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input format is not supported for fast tilize")

    input_dimensions = [input_height * 32, input_width * 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(TilizeGolden)
    golden_tensor = generate_golden(src_A, input_dimensions, formats.output)

    configuration = TestConfig(
        "sources/fast_tilize_test.cpp",
        formats,
        templates=[INPUT_DIMENSIONS(input_dimensions, input_dimensions)],
        runtimes=[TILE_COUNT(tile_cnt_A), LOOP_FACTOR(1), NUM_FACES(4)],
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
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output])

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
