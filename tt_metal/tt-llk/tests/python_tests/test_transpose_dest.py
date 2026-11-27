# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, Transpose
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
from helpers.utils import passed_test, result_tensor

TRANSPOSE_DEST_FLOAT_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)


def transpose_dest_float_dest_acc(formats):
    """16-bit inputs run with both a 16-bit and a 32-bit dest; 32-bit inputs need a 32-bit dest."""
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def transpose_dest_float_math_transpose_faces(formats):
    """Transpose of 16-bit values in dest is supported only for math_transpose_faces = Yes."""
    if formats.input_format.is_32_bit():
        return [Transpose.Yes, Transpose.No]
    return [Transpose.Yes]


def transpose_dest_float_unpack_to_dest(formats, math_transpose_faces):
    """Lossy (unpack to src regs) and lossless (unpack to dest) transpose for 32-bit inputs.

    Covered combinations:
    1. Lossless transpose of 32-bit values in dest -> input_format=Float32, dest_acc=Yes and unpack_to_dest=True.
    2. Transpose of 32-bit values with precision loss (Float32 unpacked to src registers truncates to Tf32) ->
       input_format=Float32, dest_acc=Yes and unpack_to_dest=False.
    3. Transpose of 32-bit values with potential precision loss -> input_format=[Bfp8_b, Float16_b],
       dest_acc=Yes and unpack_to_dest=False (intermediate data is 32-bit in dest).
    4. Transpose of 16-bit values in dest -> input_format=[Float16_b, Bfp8_b],
       dest_acc=No and unpack_to_dest=False.

    math_transpose_faces = Transpose.No transposes within faces only (32-bit dest) and always unpacks to dest.
    """
    if math_transpose_faces == Transpose.Yes:
        return [False, True] if formats.input_format.is_32_bit() else [False]
    return [True]


@parametrize(
    formats=TRANSPOSE_DEST_FLOAT_FORMATS,
    dest_acc=transpose_dest_float_dest_acc,
    math_transpose_faces=transpose_dest_float_math_transpose_faces,
    unpack_to_dest=transpose_dest_float_unpack_to_dest,
)
def test_transpose_dest_float(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest)


@parametrize(
    formats=input_output_formats([DataFormat.Int32], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes, Transpose.No],
    unpack_to_dest=[True],
)
def test_transpose_dest_int(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest)


def transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest):

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
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    configuration = TestConfig(
        "sources/transpose_dest_test.cpp",
        formats,
        templates=[
            # The kernel only syncs Dest in halves.
            DEST_SYNC(DestSync.Half),
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
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

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    if (
        unpack_to_dest == True
        and dest_acc == DestAccumulation.Yes
        and formats.input_format.is_32_bit()
        and formats.output_format.is_32_bit()
    ):
        is_equal = torch.equal(res_tensor, golden_tensor)
        assert is_equal, "Assert against golden failed"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"
