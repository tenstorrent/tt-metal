# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.profiler import ProfilerConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Int32],
    ),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    unpack_transpose_within_face=[Transpose.No, Transpose.Yes],
)
def test_perf_unpack_transpose(
    perf_report,
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
    workers_tensix_coordinates,
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

    tile_count = 16

    configuration = ProfilerConfig(
        "sources/unpack_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE],
        templates=[],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(unpack_transpose_within_face),
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
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
