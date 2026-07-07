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
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM

# Element-space geometries: every rt x ct grid of 32x32 tiles. The tile count is
# derived from input_dimensions in the body, so there is a single geometry axis
# (input_dimensions) rather than a separate tile-grid axis.
_UNPACK_TILIZE_FLOAT_INPUT_DIMENSIONS = [
    [rt * DEFAULT_TILE_R_DIM, ct * DEFAULT_TILE_C_DIM]
    for rt in range(1, 9)
    for ct in range(1, 9)
]
_UNPACK_TILIZE_INT_INPUT_DIMENSIONS = [
    [rt * DEFAULT_TILE_R_DIM, ct * DEFAULT_TILE_C_DIM]
    for rt in range(1, 3)
    for ct in range(1, 3)
]


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
    input_dimensions=_UNPACK_TILIZE_FLOAT_INPUT_DIMENSIONS,
)
def test_perf_unpack_tilize_float(
    perf_report,
    formats,
    num_faces,
    input_dimensions,
):
    if formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b input not supported for unpack_tilize")

    _perf_unpack_tilize(
        perf_report,
        formats,
        num_faces,
        input_dimensions,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_faces=[4],
    input_dimensions=_UNPACK_TILIZE_INT_INPUT_DIMENSIONS,
)
def test_perf_unpack_tilize_int(
    perf_report,
    formats,
    num_faces,
    input_dimensions,
):
    _perf_unpack_tilize(
        perf_report,
        formats,
        num_faces,
        input_dimensions,
    )


def _perf_unpack_tilize(
    perf_report,
    formats,
    num_faces,
    input_dimensions,
):
    assert (
        input_dimensions[0] % DEFAULT_TILE_R_DIM == 0
        and input_dimensions[1] % DEFAULT_TILE_C_DIM == 0
    ), f"input_dimensions {input_dimensions} must be a whole number of 32x32 tiles"
    tile_count = (input_dimensions[0] // DEFAULT_TILE_R_DIM) * (
        input_dimensions[1] // DEFAULT_TILE_C_DIM
    )

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
