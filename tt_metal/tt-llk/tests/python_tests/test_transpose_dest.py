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
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType, Transpose
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
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


TRANSPOSE_DEST_FLOAT_SWEEP = dict(
    formats=TRANSPOSE_DEST_FLOAT_FORMATS,
    dest_acc=transpose_dest_float_dest_acc,
    math_transpose_faces=transpose_dest_float_math_transpose_faces,
    unpack_to_dest=transpose_dest_float_unpack_to_dest,
)

TRANSPOSE_DEST_FLOAT_PERF_SWEEP = {
    **TRANSPOSE_DEST_FLOAT_SWEEP,
    "formats": input_output_formats([DataFormat.Float16_b]),
}

TRANSPOSE_DEST_INT_SWEEP = dict(
    formats=input_output_formats([DataFormat.Int32], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes, Transpose.No],
    unpack_to_dest=[True],
)


@parametrize(**TRANSPOSE_DEST_FLOAT_SWEEP)
def test_transpose_dest_float(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    _run_transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest)


@parametrize(**TRANSPOSE_DEST_INT_SWEEP)
def test_transpose_dest_int(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
):
    _run_transpose_dest(formats, dest_acc, math_transpose_faces, unpack_to_dest)


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_dest_int8(
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
    transpose_dest_int8(formats, dest_acc, math_transpose_faces, unpack_to_dest)


@parametrize(
    formats=input_output_formats([DataFormat.Int8], same=True),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes],
    unpack_to_dest=[False],
)
def test_transpose_dest_int8_single_tile(
    formats,
    dest_acc,
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
    )


def transpose_dest_int8(
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


def _run_transpose_dest(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    # Exercise four destination-register usages. FP32 destination modes hold four
    # tiles per half, while 16-bit modes hold eight tiles per half.
    input_dimensions = (
        [128, 128]
        if dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit()
        else [256, 128]
    )

    src_A = src_B = None
    golden_tensor = None
    if is_perf:
        tile_cnt_A = tile_cnt_B = (
            input_dimensions[0]
            * input_dimensions[1]
            // (TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1])
        )
    else:
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_dimensions,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=input_dimensions,
        )

        # Generate custom test input stimuli to check if zeroflag fix works.
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

    unpack_transpose_faces = (
        Transpose.Yes
        if dest_acc == DestAccumulation.Yes and math_transpose_faces == Transpose.No
        else Transpose.No
    )

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=dict(
            test_name="sources/transpose_dest_test.cpp",
            formats=formats,
            templates=[
                # The kernel only syncs Dest in halves.
                DEST_SYNC(DestSync.Half),
                MATH_TRANSPOSE_FACES(math_transpose_faces),
            ],
            runtimes=[
                UNPACK_TRANS_FACES(unpack_transpose_faces),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(),
                NUM_BLOCKS(num_blocks),
                NUM_TILES_IN_BLOCK(num_tiles_in_block),
                LOOP_FACTOR(loop_factor),
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
        ),
    )

    if is_perf:
        configuration.run(perf_report)
        return

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
