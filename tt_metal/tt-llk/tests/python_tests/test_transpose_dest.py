# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.dest_params import (
    dest_acc_modes,
    dest_sync_modes,
)
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, Transpose, format_dict
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_TRANSPOSE_FACES,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.utils import passed_test

TRANSPOSE_DEST_FLOAT_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)


def transpose_dest_dest_acc(formats):
    if formats.input_format.is_32_bit() or is_dest_acc_needed(formats):
        return dest_acc_modes(formats, allowed=[DestAccumulation.Yes])
    return dest_acc_modes(formats, allowed=[DestAccumulation.No])


def transpose_dest_math_transpose_faces(formats):
    return (
        [Transpose.Yes, Transpose.No]
        if formats.input_format.is_32_bit()
        else [Transpose.Yes]
    )


def transpose_dest_unpack_to_dest(formats, math_transpose_faces):
    if math_transpose_faces == Transpose.Yes:
        return [False, True] if formats.input_format.is_32_bit() else [False]
    return [True]


@parametrize(
    formats=TRANSPOSE_DEST_FLOAT_FORMATS,
    dest_acc=transpose_dest_dest_acc,
    dest_sync=lambda: dest_sync_modes(),
    math_transpose_faces=transpose_dest_math_transpose_faces,
    unpack_to_dest=transpose_dest_unpack_to_dest,
)
def test_transpose_dest_float(
    formats,
    dest_acc,
    dest_sync,
    math_transpose_faces,
    unpack_to_dest,
):
    transpose_dest(
        formats=formats,
        dest_acc=dest_acc,
        dest_sync=dest_sync,
        math_transpose_faces=math_transpose_faces,
        unpack_to_dest=unpack_to_dest,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    dest_acc=[DestAccumulation.Yes],
    dest_sync=lambda: dest_sync_modes(),
    math_transpose_faces=[Transpose.Yes, Transpose.No],
    unpack_to_dest=[True],
)
def test_transpose_dest_int(
    formats,
    dest_acc,
    dest_sync,
    math_transpose_faces,
    unpack_to_dest,
):
    transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest, dest_sync)


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    dest_sync=lambda: dest_sync_modes(),
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_dest_int8(
    formats,
    dest_acc,
    dest_sync,
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
    transpose_dest_int8(
        formats, dest_acc, math_transpose_faces, unpack_to_dest, dest_sync=dest_sync
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    dest_sync=lambda: dest_sync_modes(),
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_dest_int8_single_tile(
    formats,
    dest_acc,
    dest_sync,
    math_transpose_faces,
    unpack_to_dest,
):
    """
    Single-tile (32x32) variant of test_transpose_dest_int8.

    Exercises the Int8 haloize unpack path on a single tile (tile_cnt == 1),
    covering the face / within-face 16x16 transpose edge case where the whole
    transpose fits in one tile rather than spanning a multi-tile grid.
    """
    transpose_dest_int8(
        formats,
        dest_acc,
        math_transpose_faces,
        unpack_to_dest,
        input_dimensions=[32, 32],
        dest_sync=dest_sync,
    )


def transpose_dest_int8(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    input_dimensions=None,
    dest_sync=DestSync.Half,
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
        templates=[MATH_TRANSPOSE_FACES(math_transpose_faces), DEST_SYNC(dest_sync)],
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

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert torch.equal(res_tensor, golden_tensor), "Assert against golden failed"


def transpose_dest(
    formats, dest_acc, math_transpose_faces, unpack_to_dest, dest_sync=DestSync.Half
):

    # Exercise four destination-register usages. FP32 destination modes hold four
    # tiles per half, while 16-bit modes hold eight tiles per half.
    input_dimensions = (
        [128, 128]
        if dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit()
        else [256, 128]
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Generate custom test input stimuli to check if zeroflag fix works
    if formats.input_format == DataFormat.Int32:
        src_A = (torch.arange(0, src_A.numel()) * 10000).reshape_as(src_A)
        src_B = (torch.arange(0, src_B.numel()) * 10000).reshape_as(src_B)

    generate_datacopy_golden = get_golden_generator(DataCopyGolden)
    datacopy_tensor = generate_datacopy_golden(
        src_A, formats.output_format, num_faces=4, input_dimensions=input_dimensions
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        t_matrix = get_golden_generator(TransposeGolden)
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            datacopy_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = t_matrix.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    else:
        golden_tensor = []

    # Partition the tiles into destination-register banks. transpose is applied
    # per tile, so the block size only controls how many tiles share a DEST bank
    # before it is packed out and reused.
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    configuration = TestConfig(
        "sources/transpose_dest_test.cpp",
        formats,
        templates=[MATH_TRANSPOSE_FACES(math_transpose_faces), DEST_SYNC(dest_sync)],
        runtimes=[
            # When math_transpose_faces is False, unpack_transpose_faces should be Transpose.Yes
            # This mode is supported only for 32-bit dest
            UNPACK_TRANS_FACES(
                Transpose.Yes
                if (
                    dest_acc == DestAccumulation.Yes
                    and math_transpose_faces == Transpose.No
                )
                else Transpose.No
            ),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
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

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if (
        unpack_to_dest == True
        and dest_acc == DestAccumulation.Yes
        and formats.input_format in (DataFormat.Int32, DataFormat.Float32)
        and formats.output_format in (DataFormat.Int32, DataFormat.Float32)
    ):
        is_equal = torch.equal(res_tensor, golden_tensor)
        assert is_equal, "Assert against golden failed"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"
