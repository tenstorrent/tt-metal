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
from helpers.perf import PerfConfig
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
    dest_acc=lambda formats: [
        (
            DestAccumulation.Yes
            if formats.input_format.is_32_bit()
            else DestAccumulation.No
        )
    ],
    unpack_to_dest=lambda formats: [formats.input_format.is_32_bit()],
    unpack_transpose_faces=[Transpose.No, Transpose.Yes],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
)
def test_perf_transpose_dest_float(
    perf_report,
    formats,
    dest_acc,
    unpack_to_dest,
    unpack_transpose_faces,
    math_transpose_faces,
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
