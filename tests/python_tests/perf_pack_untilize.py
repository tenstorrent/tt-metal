# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfRunType, perf_benchmark, update_report


@pytest.mark.perf
@parametrize(
    test_name="pack_untilize_perf",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]
    ),
    full_rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    full_ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_pack_untilize(perf_report, test_name, formats, full_rt_dim, full_ct_dim):
    if formats.output_format == DataFormat.Bfp8_b:
        pytest.skip("Pack Untilize does not support Bfp8_b output")

    if (formats.input_format == DataFormat.Int32) ^ (
        formats.output_format == DataFormat.Int32
    ):
        pytest.skip("Pack Untilize does not support mixing Int32 with other formats")

    max_block_dim = 4 if formats.input_format.is_32_bit() else 8

    # fixme: handle format outlier case properly
    if (
        formats.input_format == DataFormat.Float16_b
        or formats.input_format == DataFormat.Bfp8_b
    ) and formats.output_format == DataFormat.Float16:
        max_block_dim = 4

    # Find the maximum block size that divides full_ct_dim and is <= max_block_dim
    block_ct_dim = 1
    for candidate in range(min(full_ct_dim, max_block_dim), 0, -1):
        if full_ct_dim % candidate == 0:
            block_ct_dim = candidate
            break

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    tile_count = full_rt_dim * full_ct_dim
    dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    test_config = {
        "formats": formats,
        "testname": test_name,
        "tile_cnt": tile_count,
        "input_A_dimensions": dimensions,
        "input_B_dimensions": dimensions,
        "block_ct_dim": block_ct_dim,
        "unpack_to_dest": formats.input_format.is_32_bit(),
    }

    results = perf_benchmark(test_config, run_types)
    update_report(perf_report, test_config, results)
