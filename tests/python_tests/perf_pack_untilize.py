# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.profiler import ProfilerConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    LOOP_FACTOR,
    TILE_COUNT,
)


@pytest.mark.perf
@parametrize(
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
def test_perf_pack_untilize(
    perf_report,
    formats,
    full_rt_dim,
    full_ct_dim,
    workers_tensix_coordinates,
):
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

    tile_count = full_rt_dim * full_ct_dim
    dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    configuration = ProfilerConfig(
        "sources/pack_untilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[INPUT_DIMENSIONS(dimensions, dimensions, block_ct_dim)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR()],
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
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
