# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report


@pytest.mark.perf
@parametrize(
    test_name="unpack_untilize_perf",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            # DataFormat.Bfp8_b, # sstanisic FIXME: add Bfp8_b support
        ]
    ),
    full_rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    full_ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_untilize(
    perf_report, test_name, formats, full_rt_dim, full_ct_dim
):

    run_types = [
        PerfRunType.L1_TO_L1,
    ]

    tile_count = full_rt_dim * full_ct_dim
    dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_count,
        "input_A_dimensions": dimensions,
        "input_B_dimensions": dimensions,
    }

    results = perf_benchmark(test_config, run_types)
    update_report(perf_report, test_config, results)
