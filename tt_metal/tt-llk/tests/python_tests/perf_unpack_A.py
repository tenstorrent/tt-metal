# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf coverage for unpack-A transpose axes (subset of test_unpack_comprehensive)."""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    PerfRunType,
    Transpose,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM

# NOTE: This perf kernel (unpack_transpose_perf.cpp) exercises only the
# unpack-A transpose path, so the only axes it can honor are the two transpose
# flags plus the input geometry. The broadcast / acc_to_dest / reuse_dest /
# stochastic_rnd / num_faces / face_r_dim axes swept by the functional
# test_unpack_comprehensive are intentionally omitted here rather than pinned to
# unused single values, so compare_test_and_perf.py reports them as
# functional-only instead of falsely identical.


@pytest.mark.perf
@parametrize(
    cpp_source=["sources/unpack_transpose_perf.cpp"],
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Int32],
    ),
    transpose_of_faces=[Transpose.No, Transpose.Yes],
    within_face_16x16_transpose=[Transpose.No, Transpose.Yes],
    input_dimensions=[[256, 256]],
)
def test_perf_unpack_comprehensive(
    perf_report,
    cpp_source,
    formats,
    transpose_of_faces,
    within_face_16x16_transpose,
    input_dimensions,
):
    # Int32 format restrictions
    if formats.input_format == DataFormat.Int32:
        # Unpacker: Int32 can ONLY unpack to Int32 (identity) in Dst register per ISA specification
        if formats.output_format != DataFormat.Int32:
            pytest.skip(
                f"Int32 -> {formats.output_format.name} conversion not supported (unpacker limitation)"
            )
        # Transpose: Int32 does not support any transposition operations
        if (
            transpose_of_faces == Transpose.Yes
            or within_face_16x16_transpose == Transpose.Yes
        ):
            pytest.skip("Transpose not supported for Int32")

    # Packer: Bfp8_b and Float16 cannot convert to Int32 in this test matrix.
    if formats.output_format == DataFormat.Int32 and formats.input_format in [
        DataFormat.Bfp8_b,
        DataFormat.Float16,
    ]:
        pytest.skip(
            f"{formats.input_format.name} -> Int32 conversion not supported (packer limitation)"
        )

    if (
        transpose_of_faces == Transpose.No
        and within_face_16x16_transpose == Transpose.No
    ):
        pytest.skip(
            "Skipping test for transpose_of_faces=False and within_face_16x16_transpose=False"
        )

    # Derive the measured tile count from the declared geometry (full 32x32
    # tiles) instead of hardcoding it, so input_dimensions is truthful.
    assert (
        input_dimensions[0] % DEFAULT_TILE_R_DIM == 0
        and input_dimensions[1] % DEFAULT_TILE_C_DIM == 0
    ), f"input_dimensions {input_dimensions} must be a whole number of 32x32 tiles"
    tile_count = (input_dimensions[0] // DEFAULT_TILE_R_DIM) * (
        input_dimensions[1] // DEFAULT_TILE_C_DIM
    )

    configuration = PerfConfig(
        cpp_source,
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE],
        templates=[],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(transpose_of_faces),
            UNPACK_TRANS_WITHIN_FACE(within_face_16x16_transpose),
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

    configuration.run(perf_report)
