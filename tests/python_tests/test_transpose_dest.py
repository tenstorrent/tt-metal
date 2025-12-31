# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, Transpose, format_dict
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_TRANSPOSE_FACES,
    NUM_FACES,
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


def generate_transpose_dest_float_combinations(formats_list):
    """
    Generate transpose dest combinations that respect constraints.

    Key rules:
    1. math_transpose_faces = Transpose.No is not supported for 16-bit dest.
    2. math_transpose_faces = Transpose.No and 32-bit dest will transpose within faces, and can be combined with
       unpack_transpose_faces = Transpose.Yes in order to transpose faces.
    3. math_transpose_faces = Transpose.Yes and 16-bit dest is supported.
    1. math_transpose_faces = Transpose.No and 32-bit dest is supported.

    Covered combinations:
    1. Lossless transpose of 32-bit values in dest -> input_format=Float32, dest_acc=DestAccumulation.Yes and unpack_to_dest=True.
    2. Transpose of 32-bit values in dest with precision loss (unpacks Float32 to src registers, Float32 truncates to Tf32) ->
       input_format=Float32, dest_acc=DestAccumulation.Yes and unpack_to_dest=False.
    3. Transpose of 16-bit values in dest -> input_format=[Float16, Float16_b, Bfp8_b],
       dest_acc=DestAccumulation.No and unpack_to_dest=False.

    Args:
        formats_list: List of InputOutputFormat combinations
    Returns:
        List of tuples: (format, dest_acc, math_transpose_faces, unpack_to_dest)
    """

    combinations = []

    for fmt in formats_list:
        is_input_32bit = fmt.input_format.is_32_bit()
        dest_acc_list = (
            [DestAccumulation.Yes]
            if is_input_32bit or is_dest_acc_needed(fmt)
            else [DestAccumulation.No]
        )

        # Transpose of 16-bit values in dest is supported only for math_transpose_faces = True
        math_transpose_faces_list = (
            [Transpose.Yes, Transpose.No] if is_input_32bit else [Transpose.Yes]
        )

        for dest_acc, math_transpose_faces in product(
            dest_acc_list, math_transpose_faces_list
        ):
            # Test both loss (unpacking to src registers) and lossless (unpacking to dest) transpose dest
            # for 32bit inputs when math_transpose_faces = Transpose.Yes
            if math_transpose_faces == Transpose.Yes:
                unpack_to_dest_list = [False, True] if is_input_32bit else [False]
            else:
                unpack_to_dest_list = [True]

            combinations.extend(
                (fmt, dest_acc, math_transpose_faces, unpack_to_dest)
                for unpack_to_dest in unpack_to_dest_list
            )

    return combinations


@parametrize(
    fmt_dest_acc_math_transp_unpack_to_dest=generate_transpose_dest_float_combinations(
        TRANSPOSE_DEST_FLOAT_FORMATS
    ),
)
def test_transpose_dest_float(
    fmt_dest_acc_math_transp_unpack_to_dest, workers_tensix_coordinates
):

    fmt_dest_acc_math_transp_unpack_to_dest = fmt_dest_acc_math_transp_unpack_to_dest[0]

    transpose_dest(
        formats=fmt_dest_acc_math_transp_unpack_to_dest[0],
        dest_acc=fmt_dest_acc_math_transp_unpack_to_dest[1],
        math_transpose_faces=fmt_dest_acc_math_transp_unpack_to_dest[2],
        unpack_to_dest=fmt_dest_acc_math_transp_unpack_to_dest[3],
        workers_tensix_coordinates=workers_tensix_coordinates,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes, Transpose.No],
    unpack_to_dest=[True],
)
def test_transpose_dest_int(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    workers_tensix_coordinates,
):
    transpose_dest(
        formats,
        dest_acc,
        math_transpose_faces,
        unpack_to_dest,
        workers_tensix_coordinates,
    )


def transpose_dest(
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    workers_tensix_coordinates,
):

    if dest_acc == DestAccumulation.Yes and formats.input_format != DataFormat.Int32:
        pytest.skip("32-bit dest tests fail for Float formats due to bit No.11 issue.")

    input_dimensions = [64, 64]

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

    configuration = TestConfig(
        "sources/transpose_dest_test.cpp",
        formats,
        templates=[MATH_TRANSPOSE_FACES(math_transpose_faces)],
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

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
