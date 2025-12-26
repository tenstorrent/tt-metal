# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    PerfRunType,
    Transpose,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.profiler import ProfilerConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATH_TRANSPOSE_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Int32],
    ),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
)
def test_perf_math_transpose(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
    workers_tensix_coordinates,
):
    if formats.input_format != formats.output_format:
        pytest.skip("Prevent mixing INT and FP in math transpose")

    if math_transpose_faces == Transpose.No and not formats.input_format.is_32_bit():
        pytest.skip(
            "Unsupported config transpose_of_faces = false and is_32bit = false"
        )

    if (
        unpack_transpose_faces == Transpose.Yes
        and math_transpose_faces == Transpose.Yes
    ):
        pytest.skip("Skip transposing faces twice")

    tile_count = 16

    configuration = ProfilerConfig(
        "sources/math_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=(
            DestAccumulation.Yes
            if formats.input_format.is_32_bit()
            else DestAccumulation.No
        ),
    )

    configuration.run(perf_report, location=workers_tensix_coordinates)
