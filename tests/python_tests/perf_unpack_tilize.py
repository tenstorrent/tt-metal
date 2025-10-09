# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report


@pytest.mark.perf
@parametrize(
    test_name="unpack_tilize_perf",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ]
    ),
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_tilize_float(perf_report, test_name, formats, rt_dim, ct_dim):
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    _perf_unpack_tilize(perf_report, test_name, formats, rt_dim, ct_dim)


@pytest.mark.perf
@parametrize(
    test_name="unpack_tilize_perf",
    formats=input_output_formats([DataFormat.Int32]),
    rt_dim=[1, 2],
    ct_dim=[1, 2],
)
def test_perf_unpack_tilize_int(perf_report, test_name, formats, rt_dim, ct_dim):
    _perf_unpack_tilize(perf_report, test_name, formats, rt_dim, ct_dim)


def _perf_unpack_tilize(perf_report, test_name, formats, rt_dim, ct_dim):
    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    tile_count = rt_dim * ct_dim
    dimensions = [rt_dim * 32, ct_dim * 32]

    test_config = {
        "formats": formats,
        "testname": test_name,
        "loop_factor": 4,
        "tile_cnt": tile_count,
        "input_A_dimensions": dimensions,
        "input_B_dimensions": dimensions,
        "unpack_to_dest": formats.input_format == DataFormat.Int32,
    }

    results = perf_benchmark(test_config, run_types)
    update_report(perf_report, test_config, results)
