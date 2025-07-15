# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from conftest import skip_for_blackhole
from helpers.device import write_stimuli_to_l1
from helpers.format_arg_mapping import DestAccumulation
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.perf import (
    PerfReport,
    PerfRunType,
    delete_benchmark_dir,
    dump_report,
    dump_scatter,
    perf_benchmark,
    update_report,
)
from helpers.stimuli_generator import generate_stimuli

TEST_NAME = "fast_tilize_test"


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


report = PerfReport()


@pytest.fixture(scope="module")
def report_fixture():
    delete_benchmark_dir(TEST_NAME)
    yield
    dump_report(TEST_NAME, report)
    dump_scatter(TEST_NAME, report)


@skip_for_blackhole
@pytest.mark.perf
@pytest.mark.parametrize("input_format", [DataFormat.Float32, DataFormat.Float16_b])
@pytest.mark.parametrize(
    "output_format", [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Bfp8_b]
)
@pytest.mark.parametrize("fp32_dest", [DestAccumulation.Yes, DestAccumulation.No])
@pytest.mark.parametrize("input_width, input_height", generate_input_dimensions(16))
def test_fast_tilize_perf(
    report_fixture, input_format, output_format, fp32_dest, input_width, input_height
):

    input_dimensions = [input_height * 32, input_width * 32]

    src_A, src_B, tile_cnt = generate_stimuli(
        input_format, input_format, input_dimensions=input_dimensions
    )

    res_address = write_stimuli_to_l1(
        src_A, src_B, input_format, input_format, tile_count=tile_cnt
    )

    formats = InputOutputFormat(input_format, output_format)

    test_config = {
        "formats": formats,
        "testname": TEST_NAME,
        "tile_cnt": input_height * input_width,
        "input_dimensions": input_dimensions,
        "dest_acc": fp32_dest,
    }

    results = perf_benchmark(test_config, [PerfRunType.L1_TO_L1], 2)
    update_report(report, test_config, results)
