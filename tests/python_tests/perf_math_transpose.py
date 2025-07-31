# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from helpers.format_arg_mapping import DestAccumulation
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import (
    PerfReport,
    PerfRunType,
    delete_benchmark_dir,
    dump_report,
    dump_scatter,
    perf_benchmark,
    update_report,
)

TEST_NAME = "math_transpose_perf"


report = PerfReport()


@pytest.fixture(scope="module")
def report_fixture():
    delete_benchmark_dir(TEST_NAME)
    yield
    dump_report(TEST_NAME, report)
    dump_scatter(TEST_NAME, report)


@pytest.mark.perf
@parametrize(
    test_name=TEST_NAME,
    formats=input_output_formats(
        [
            # DataFormat.Float16_b,     # Waiting for resolution of issue 549
            DataFormat.Int32
        ],
    ),
    unpack_transpose_faces=[False, True],
    math_transpose_faces=[False, True],
)
def test_perf_math_transpose(
    report_fixture,
    test_name,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
):

    if not math_transpose_faces and not formats.input_format.is_32_bit():
        pytest.skip(
            "Unsupported config transpose_of_faces = false and is_32bit = false"
        )

    if unpack_transpose_faces and math_transpose_faces:
        pytest.skip("Skip transposing faces twice")

    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )

    test_config = {
        "testname": test_name,
        "formats": formats,
        "tile_cnt": 16,
        "dest_acc": dest_acc,
        "unpack_to_dest": formats.input_format.is_32_bit(),
        "unpack_transpose_faces": unpack_transpose_faces,
        "math_transpose_faces": math_transpose_faces,
    }

    results = perf_benchmark(test_config, run_types=[PerfRunType.L1_TO_L1])
    update_report(report, test_config, results)
