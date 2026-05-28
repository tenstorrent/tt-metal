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
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
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
    UNPACKER_ENGINE_SEL,
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
        List of (format, dest_acc, input_dimensions, relu_type) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int8, UInt8 require 32bit mode dest register for Eltwise binary operations
        if in_fmt == DataFormat.Int8 or in_fmt == DataFormat.UInt8:
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    dimensions_cache = {
        (dest_acc, dest_sync): tuple(
            generate_unary_input_dimensions(dest_acc, dest_sync)
        )
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    combinations = []

    for fmt in formats_list:
        for acc in get_dest_acc_modes(fmt.input_format):
            for dest_sync in (DestSync.Half, DestSync.Full):
                for unp_tilize_sel in (UnpackerEngine.UnpA, UnpackerEngine.UnpB):
                    for dimensions in dimensions_cache[(acc, dest_sync)]:
                        combinations.append(
                            (fmt, acc, dest_sync, unp_tilize_sel, dimensions)
                        )

    return combinations


def generate_unpack_tilize_binary_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_tilize_binary_operands test combinations.

    Only non-32-bit formats are supported for now.

    Returns: List of (format, dest_acc, dest_sync, unp_tilize_sel, input_dimensions) tuples
    """
    dest_acc = [DestAccumulation.Yes, DestAccumulation.No]

    dimensions_cache = {
        (acc, dest_sync): tuple(generate_unary_input_dimensions(acc, dest_sync))
        for acc in dest_acc
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    combinations = []

    for fmt in formats_list:
        if fmt.input_format.is_32_bit():
            continue

        for acc in dest_acc:
            for dest_sync in (DestSync.Half, DestSync.Full):
                for unp_tilize_sel in (UnpackerEngine.UnpA, UnpackerEngine.UnpB):
                    for dimensions in dimensions_cache[(acc, dest_sync)]:
                        combinations.append(
                            (fmt, acc, dest_sync, unp_tilize_sel, dimensions)
                        )

    return combinations


UNPACK_TILIZE_BINARY_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8P,
        DataFormat.MxFp8R,
        DataFormat.MxFp4,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Float16_b,
        DataFormat.Float16,
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

    num_faces = 4

    tilize_a = unp_tilize_sel == UnpackerEngine.UnpA

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    tilize_gen = get_golden_generator(TilizeGolden)
    eltwise_binary_gen = get_golden_generator(EltwiseBinaryGolden)

    tilize_src = src_A if tilize_a else src_B
    tilized = tilize_gen(
        tilize_src, input_dimensions, formats.input_format, num_faces=num_faces
    )

    golden_tensor = eltwise_binary_gen(
        MathOperation.Elwadd,
        tilized if tilize_a else src_A,
        src_B if tilize_a else tilized,
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
            UNPACKER_ENGINE_SEL(unp_tilize_sel),
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
