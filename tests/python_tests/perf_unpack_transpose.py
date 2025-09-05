# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from helpers.format_arg_mapping import Transpose
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import (
    PerfRunType,
    perf_benchmark,
    update_report,
)


@pytest.mark.perf
@parametrize(
    test_name="unpack_transpose_perf",
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Int32],
    ),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    unpack_transpose_within_face=[Transpose.No, Transpose.Yes],
)
def test_perf_unpack_transpose(
    perf_report,
    test_name,
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
):

    if (
        formats.input == DataFormat.Int32
        and unpack_transpose_within_face == Transpose.Yes
    ):
        pytest.skip("Unpack within face not supported for Int32")

    if (
        unpack_transpose_faces == Transpose.No
        and unpack_transpose_within_face == Transpose.No
    ):
        pytest.skip(
            "Skipping test for unpack_transpose_faces=False and unpack_transpose_within_face=False"
        )

    test_config = {
        "testname": test_name,
        "formats": formats,
        "tile_cnt": 16,
        "unpack_transpose_faces": unpack_transpose_faces,
        "unpack_transpose_within_face": unpack_transpose_within_face,
    }

    results = perf_benchmark(
        test_config, run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE]
    )
    update_report(perf_report, test_config, results)
