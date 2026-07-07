# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

_UNPACK_TILIZE_FLOAT_DIMENSIONS = [(rt, ct) for rt in range(1, 9) for ct in range(1, 9)]
_UNPACK_TILIZE_INT_DIMENSIONS = [(rt, ct) for rt in range(1, 3) for ct in range(1, 3)]


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
    num_faces=[4],
    dimensions=_UNPACK_TILIZE_FLOAT_DIMENSIONS,
    input_dimensions=lambda dimensions: [[dimensions[0] * 32, dimensions[1] * 32]],
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    num_faces,
    dimensions,
    input_dimensions,
):
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    rt_dim, ct_dim = dimensions
    _perf_unpack_tilize(
        perf_report,
        formats,
        num_faces,
        input_dimensions,
        rt_dim,
        ct_dim,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_faces=[4],
    dimensions=_UNPACK_TILIZE_INT_DIMENSIONS,
    input_dimensions=lambda dimensions: [[dimensions[0] * 32, dimensions[1] * 32]],
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    num_faces,
    dimensions,
    input_dimensions,
):
    rt_dim, ct_dim = dimensions
    _perf_unpack_tilize(
        perf_report,
        formats,
        num_faces,
        input_dimensions,
        rt_dim,
        ct_dim,
    )


def _perf_unpack_tilize(
    perf_report,
    formats,
    num_faces,
    input_dimensions,
    rt_dim,
    ct_dim,
):
    assert input_dimensions == [rt_dim * 32, ct_dim * 32]
    tile_count = rt_dim * ct_dim

    configuration = PerfConfig(
        "sources/unpack_tilize_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_count),
            LOOP_FACTOR(4),
            NUM_FACES(num_faces),
        ],
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

    configuration.run(perf_report)
