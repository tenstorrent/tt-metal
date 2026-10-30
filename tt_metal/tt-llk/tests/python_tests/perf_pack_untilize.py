# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from test_pack_untilize import test_pack_untilize as run_pack_untilize


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
            DataFormat.Fp8_e4m3,
        ]
    ),
    full_rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    full_ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    run_types=[
        [
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ]
    ],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_pack_untilize(
    perf_report,
    formats,
    full_rt_dim,
    full_ct_dim,
    run_types,
    loop_factor,
    is_perf,
):
    if (
        formats.input_format == DataFormat.Fp8_e4m3
        or formats.output_format == DataFormat.Fp8_e4m3
    ) and get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "Pack Untilize does not support Fp8_e4m3 format on non-BLACKHOLE architectures"
        )

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

    dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    run_pack_untilize(
        formats,
        DestAccumulation.No,
        dimensions,
        DestSync.Half,
        0,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        block_ct_dim=block_ct_dim,
    )
