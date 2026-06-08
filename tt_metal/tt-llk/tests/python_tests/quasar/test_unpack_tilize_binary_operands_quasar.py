# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    TilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    TilizeUnpackerSel,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TILIZE_UNPACKER_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test


def generate_unpack_tilize_binary_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_tilize_binary_operands test combinations for Quasar.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, dest_sync, tilize_unpacker_sel, input_dimensions) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        # Unpack to dest is not supported for unpack tilize binary, there input cannot be Int32
        if in_fmt == DataFormat.Int32:
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int8, UInt8 require 32bit mode dest register for Eltwise binary operations
        if in_fmt == DataFormat.Int8 or in_fmt == DataFormat.UInt8:
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    def is_supported_dest_mode_dependent_conversion(out_fmt, dest_acc):
        """Check if the format conversion is supported by packer. These format conversions are dependent on the dest register mode."""
        # Upcasting to Float32/Int32 requires dest_acc enabled
        if out_fmt.is_32_bit() and dest_acc == DestAccumulation.No:
            return False
        return True

    # Targeted dimensions per (dest_sync, dest_acc) that cover key corner cases:
    # 1 tile (minimum), max-wide (stresses block_ct), max-tall (stresses block_rt),
    # and max-square (both loops at capacity).
    tilize_binary_dims = {
        (DestSync.Half, DestAccumulation.No): [
            [32, 32],
            [32, 256],
            [256, 32],
            [64, 128],
        ],
        (DestSync.Half, DestAccumulation.Yes): [
            [32, 32],
            [32, 128],
            [128, 32],
            [64, 64],
        ],
        (DestSync.Full, DestAccumulation.No): [
            [32, 32],
            [32, 512],
            [512, 32],
            [128, 128],
        ],
        (DestSync.Full, DestAccumulation.Yes): [
            [32, 32],
            [32, 256],
            [256, 32],
            [64, 128],
        ],
    }

    combinations = []

    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue
        for acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(out_fmt, acc):
                for dest_sync in (DestSync.Half, DestSync.Full):
                    for unp_tilize_sel in (
                        TilizeUnpackerSel.UnpA,
                        TilizeUnpackerSel.UnpB,
                        TilizeUnpackerSel.UnpAB,
                    ):
                        for dimensions in tilize_binary_dims[(dest_sync, acc)]:
                            combinations.append(
                                (fmt, acc, dest_sync, unp_tilize_sel, dimensions)
                            )

    return combinations


UNPACK_TILIZE_BINARY_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8P,
        DataFormat.MxFp8R,
        DataFormat.MxFp4,
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Int32,
    ],
)
ALL_UNPACK_TILIZE_BINARY_OPERANDS_COMBINATIONS = (
    generate_unpack_tilize_binary_combinations(UNPACK_TILIZE_BINARY_FORMATS)
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_tilize_sel_dims=ALL_UNPACK_TILIZE_BINARY_OPERANDS_COMBINATIONS,
)
def test_unpack_tilize_binary_operands_quasar(
    formats_dest_acc_sync_tilize_sel_dims, boot_mode=BootMode.DEFAULT
):
    (formats, dest_acc, dest_sync_mode, unp_tilize_sel, input_dimensions) = (
        formats_dest_acc_sync_tilize_sel_dims[0]
    )

    if formats.input_format == DataFormat.MxFp4 and (
        formats.output_format == DataFormat.Float32
        or formats.output_format == DataFormat.Float16_b
        or formats.output_format == DataFormat.Float16
        or formats.output_format == DataFormat.MxFp8P
    ):
        pytest.skip(
            "MxFp4 to Float32/Float16_b/Float16/MxFp8P conversion has rounding errors"
        )

    num_faces = 4

    tilize_a = unp_tilize_sel in (TilizeUnpackerSel.UnpA, TilizeUnpackerSel.UnpAB)
    tilize_b = unp_tilize_sel in (TilizeUnpackerSel.UnpB, TilizeUnpackerSel.UnpAB)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    tilize_gen = get_golden_generator(TilizeGolden)
    eltwise_binary_gen = get_golden_generator(EltwiseBinaryGolden)

    golden_A = (
        tilize_gen(src_A, input_dimensions, formats.input_format, num_faces=num_faces)
        if tilize_a
        else src_A
    )
    golden_B = (
        tilize_gen(src_B, input_dimensions, formats.input_format, num_faces=num_faces)
        if tilize_b
        else src_B
    )

    golden_tensor = eltwise_binary_gen(
        MathOperation.Elwadd,
        golden_A,
        golden_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format,
    )

    configuration = TestConfig(
        "sources/quasar/unpack_tilize_binary_operands_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            TILIZE_UNPACKER_SEL(unp_tilize_sel),
            MATH_OP(mathop=MathOperation.Elwadd),
            MATH_FIDELITY(MathFidelity.LoFi),
            DEST_SYNC(dest_sync_mode),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),
            NUM_FACES(),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
