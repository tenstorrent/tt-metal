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
            DataFormat.Bfp8_b,
        ]
    ),
    rt_dim=[1, 2, 3, 4, 5, 6, 7, 8],
    ct_dim=[1, 2, 3, 4, 5, 6, 7, 8],
)
def test_perf_unpack_tilize_float(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    _perf_unpack_tilize(
        perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    rt_dim=[1, 2],
    ct_dim=[1, 2],
)
def test_perf_unpack_tilize_int(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    _perf_unpack_tilize(
        perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
    )


def _perf_unpack_tilize(
    perf_report, formats, rt_dim, ct_dim, workers_tensix_coordinates
):
    tile_count = rt_dim * ct_dim
    dimensions = [rt_dim * 32, ct_dim * 32]

    configuration = ProfilerConfig(
        "sources/unpack_tilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[INPUT_DIMENSIONS(dimensions, dimensions)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(4)],
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
        unpack_to_dest=formats.input_format == DataFormat.Int32,
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
