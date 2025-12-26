# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
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
            # DataFormat.Bfp8_b, # sstanisic FIXME: add Bfp8_b support
        ]
    ),
    full_rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    full_ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_untilize(
    perf_report, formats, full_rt_dim, full_ct_dim, workers_tensix_coordinates
):
    tile_count = full_rt_dim * full_ct_dim
    input_dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    configuration = ProfilerConfig(
        "sources/unpack_untilize_perf.cpp",
        formats,
        [PerfRunType.L1_TO_L1],
        templates=[INPUT_DIMENSIONS(input_dimensions, input_dimensions)],
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
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
