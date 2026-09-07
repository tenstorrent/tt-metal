# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.dest_params import (
    UnpackPath,
    dest_acc_modes,
    dest_sync_modes,
    dest_tile_capacity,
    unpack_to_dest_modes,
)
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
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
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
    dest_acc=lambda formats: dest_acc_modes(
        formats,
        allowed=[
            (
                DestAccumulation.Yes
                if formats.input_format.is_32_bit()
                else DestAccumulation.No
            )
        ],
        distinct=True,
    ),
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Int32Dest
    ),
    tile_count=lambda dest_acc, dest_sync: dest_tile_capacity(dest_sync, dest_acc),
)
def test_perf_transpose_dest(
    perf_report,
    formats,
    unpack_transpose_faces,
    math_transpose_faces,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    tile_count,
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

    configuration = PerfConfig(
        "sources/math_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_TRANSPOSE_FACES(math_transpose_faces),
            DEST_SYNC(dest_sync),
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
