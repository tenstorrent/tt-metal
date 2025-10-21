# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
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
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
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
    test_name="transpose_dest_test",
    fmt_dest_acc_math_transp_unpack_to_dest=generate_transpose_dest_float_combinations(
        TRANSPOSE_DEST_FLOAT_FORMATS
    ),
)
def test_transpose_dest_float(test_name, fmt_dest_acc_math_transp_unpack_to_dest):

    transpose_dest(
        test_name,
        formats=fmt_dest_acc_math_transp_unpack_to_dest[0],
        dest_acc=fmt_dest_acc_math_transp_unpack_to_dest[1],
        math_transpose_faces=fmt_dest_acc_math_transp_unpack_to_dest[2],
        unpack_to_dest=fmt_dest_acc_math_transp_unpack_to_dest[3],
    )


@parametrize(
    test_name="transpose_dest_test",
    formats=input_output_formats([DataFormat.Int32]),
    dest_acc=[DestAccumulation.Yes],
    math_transpose_faces=[Transpose.Yes, Transpose.No],
    unpack_to_dest=[True],
)
def test_transpose_dest_int(
    test_name, formats, dest_acc, math_transpose_faces, unpack_to_dest
):
    transpose_dest(test_name, formats, dest_acc, math_transpose_faces, unpack_to_dest)


def transpose_dest(test_name, formats, dest_acc, math_transpose_faces, unpack_to_dest):

    if dest_acc == DestAccumulation.Yes and formats.input_format != DataFormat.Int32:
        pytest.skip("32-bit dest tests fail for Float formats due to bit No.11 issue.")

    input_dimensions = [32, 32]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
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
        num_tiles=tile_cnt,
        tilize=False,
        input_dimensions=input_dimensions,
    )
    golden_tensor = t_matrix.transpose_within_faces_multi_tile(
        golden_tensor,
        formats.output_format,
        num_tiles=tile_cnt,
        untilize=False,
        input_dimensions=input_dimensions,
    )

    # When math_transpose_faces is False, unpack_transpose_faces should be Transpose.Yes
    # This mode is supported only for 32-bit dest
    unpack_transpose_faces = (
        Transpose.Yes
        if (dest_acc == DestAccumulation.Yes and math_transpose_faces == Transpose.No)
        else Transpose.No
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
        "unpack_transpose_faces": unpack_transpose_faces,
        "math_transpose_faces": math_transpose_faces,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
