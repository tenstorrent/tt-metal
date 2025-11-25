# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    Transpose,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, run_test
from helpers.utils import passed_test


def generate_unpack_unary_operand_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_unary_operand combinations.

    Rules:
    1. When unpacking to dest, transpose is not yet supported.

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, transpose_en, unpacker_sel, input_dimensions) tuples
    """
    dimensions_cache = {
        DestAccumulation.No: tuple(
            generate_unary_input_dimensions(DestAccumulation.No)
        ),
        DestAccumulation.Yes: tuple(
            generate_unary_input_dimensions(DestAccumulation.Yes)
        ),
    }

    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        transpose_modes = (
            (Transpose.No,) if in_fmt.is_32_bit() else (Transpose.No, Transpose.Yes)
        )
        unpacker_engines = (
            (UnpackerEngine.UnpDest,)
            if in_fmt.is_32_bit()
            else (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
        )

        for dest_acc in dest_acc_modes:
            if (
                in_fmt != DataFormat.Float32
                and fmt.output_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                # Skip if input format is not Float32 and output format is Float32 and dest_acc is No
                # This combination is not supported in the Quasar Packer format conversions
                continue
            for transpose_en in transpose_modes:
                for unpacker_sel in unpacker_engines:
                    for dimensions in dimensions_cache[dest_acc]:
                        combinations.append(
                            (fmt, dest_acc, transpose_en, unpacker_sel, dimensions)
                        )

    return combinations


UNPACK_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
ALL_UNPACK_UNARY_OPERAND_COMBINATIONS = generate_unpack_unary_operand_combinations(
    UNPACK_FORMATS
)


@pytest.mark.quasar
@parametrize(
    test_name="unpack_unary_operand_quasar_test",
    formats_dest_acc_transpose_unpack_sel_dims=ALL_UNPACK_UNARY_OPERAND_COMBINATIONS,
)
def test_unpack_unary_operand_quasar(
    test_name, formats_dest_acc_transpose_unpack_sel_dims, boot_mode=BootMode.DEFAULT
):
    formats = formats_dest_acc_transpose_unpack_sel_dims[0]
    dest_acc = formats_dest_acc_transpose_unpack_sel_dims[1]
    transpose_en = formats_dest_acc_transpose_unpack_sel_dims[2]
    unpacker_sel = formats_dest_acc_transpose_unpack_sel_dims[3]
    input_dimensions = formats_dest_acc_transpose_unpack_sel_dims[4]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    golden_src = (
        src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    )  # use A for UnpA and UnpDest
    if transpose_en == Transpose.Yes:
        generate_golden = get_golden_generator(TransposeGolden)
        golden_tensor = generate_golden.transpose_faces_multi_tile(
            golden_src,
            formats.output_format,
            num_tiles=tile_cnt,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = generate_golden.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    else:
        generate_golden = get_golden_generator(DataCopyGolden)
        golden_tensor = generate_golden(
            golden_src,
            formats.output_format,
            num_faces=4,
            input_dimensions=input_dimensions,
        )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
        "unpack_transpose_faces": transpose_en,
        "unpack_transpose_within_face": transpose_en,
        "unpacker_engine_sel": unpacker_sel,
        "implied_math_format": ImpliedMathFormat.Yes,
        "data_copy_type": (
            DataCopyType.B2D
            if unpacker_sel == UnpackerEngine.UnpB
            else DataCopyType.A2D
        ),
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
        num_faces=4,
    )

    run_test(test_config, boot_mode=boot_mode)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=4
    )

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
