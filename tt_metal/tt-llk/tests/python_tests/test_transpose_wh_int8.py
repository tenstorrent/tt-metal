# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import TransposeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    MATH_TRANSPOSE_FACES,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.utils import result_tensor


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_wh_int8(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    """
    Test transpose_tile with Int8 format using the haloize unpack path.

    Int8 uses the non-unpack-to-dest path in transpose_init/transpose_tile:
    the unpacker performs transpose_of_faces + within_face_16x16_transpose (haloize), and
    the math thread uses A2D datacopy with is_int_fpu_en=true to reconstruct Int8 in DEST.
    No _llk_math_transpose_dest_ call is made — the unpacker already produced the transposed tile.
    """
    transpose_wh_int8(formats, dest_acc, math_transpose_faces, unpack_to_dest)


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_wh_int8_single_tile(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    """
    Single-tile (32x32) variant of test_transpose_wh_int8.

    Exercises the Int8 haloize unpack path on a single tile (tile_cnt == 1),
    covering the face / within-face 16x16 transpose edge case where the whole
    transpose fits in one tile rather than spanning a multi-tile grid.
    """
    transpose_wh_int8(
        formats,
        dest_acc,
        math_transpose_faces,
        unpack_to_dest,
        input_dimensions=[32, 32],
    )


def transpose_wh_int8(
    formats, dest_acc, math_transpose_faces, unpack_to_dest, input_dimensions=None
):

    if input_dimensions is None:
        input_dimensions = [64, 64]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        t_matrix = get_golden_generator(TransposeGolden)
        # The haloize unpack path performs transpose_of_faces + within_face_16x16_transpose.
        # There is no DataCopy golden step (the A2D datacopy only moves data to DEST; it does
        # not change the transpose already done by the unpacker). Generate the golden by
        # applying both transpose steps directly to the raw input.
        golden_faces = t_matrix.transpose_faces_multi_tile(
            src_A,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = t_matrix.transpose_within_faces_multi_tile(
            golden_faces,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    else:
        golden_tensor = []

    configuration = TestConfig(
        "sources/transpose_wh_int8_test.cpp",
        formats,
        templates=[MATH_TRANSPOSE_FACES(math_transpose_faces)],
        runtimes=[
            # The kernel hard-codes transpose_of_faces=1 (the haloize path); this
            # param is currently unused by transpose_wh_int8_test.cpp but is set to
            # Yes to stay aligned with the kernel's actual behavior.
            UNPACK_TRANS_FACES(Transpose.Yes),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert torch.equal(res_tensor, golden_tensor), "Assert against golden failed"
