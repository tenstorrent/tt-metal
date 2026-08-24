# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    PerfRunType,
    Transpose,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    MATH_TRANSPOSE_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)


def _run_transpose_dest_perf(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
    dest_acc,
    input_dimensions,
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

    tile_count = (input_dimensions[0] * input_dimensions[1]) // 1024

    configuration = PerfConfig(
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
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
    dest_acc=[DestAccumulation.No],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(
        dest_acc, DestSync.Half
    ),
)
def test_perf_transpose_dest_float(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
    dest_acc,
    input_dimensions,
):
    _run_transpose_dest_perf(
        perf_report,
        formats,
        unpack_transpose_faces,
        math_transpose_faces,
        dest_acc,
        input_dimensions,
    )


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(
        dest_acc, DestSync.Half
    ),
)
def test_perf_transpose_dest_int(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
    dest_acc,
    input_dimensions,
):
    _run_transpose_dest_perf(
        perf_report,
        formats,
        unpack_transpose_faces,
        math_transpose_faces,
        dest_acc,
        input_dimensions,
    )
