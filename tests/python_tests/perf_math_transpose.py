# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report


@pytest.mark.perf
@parametrize(
    test_name="math_transpose_perf",
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Int32],
    ),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
)
def test_perf_math_transpose(
    perf_report,
    test_name,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
):
    if formats.input_format != formats.output_format:
        pytest.skip("Prevent mixing INT and FP in math transpose")

    if math_transpose_faces == Transpose.No and not formats.input_format.is_32_bit():
        pytest.skip(
            "Unsupported config transpose_of_faces = false and is_32bit = false"
        )

    if (
        unpack_transpose_faces == Transpose.Yes
        and math_transpose_faces == Transpose.Yes
    ):
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
    update_report(perf_report, test_config, results)
