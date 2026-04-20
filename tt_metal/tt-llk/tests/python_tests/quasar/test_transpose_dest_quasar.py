# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
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
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_TRANSPOSE_FACES,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def generate_qsr_transpose_dest_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate transpose dest combinations for Quasar tests.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, dest_sync, math_transpose_faces) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        # Int8/UInt8 in Src regs and Int32 in dest reg is unsupported for MOVB2D
        # Float16/Float16_b in Src regs and Float32 in dest reg is unsupported for MOVB2D
        return (DestAccumulation.No,)

    def is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
        """Check if the format conversion is supported by packer. These format conversions are dependent on the dest register mode."""
        # Upcasting to Float32/Int32 requires dest_acc enabled
        if (
            out_fmt.is_32_bit()
            and not in_fmt.is_32_bit()
            and dest_acc == DestAccumulation.No
        ):
            return False
        # Int8<->UInt8 conversion requires dest_acc enabled
        if (
            dest_acc == DestAccumulation.No
            and in_fmt in (DataFormat.Int8, DataFormat.UInt8)
            and in_fmt != out_fmt
        ):
            return False
        return True

    dimensions_cache = {
        (dest_acc, dest_sync): tuple(
            generate_unary_input_dimensions(dest_acc, dest_sync)
        )
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    dest_sync_modes = (DestSync.Half, DestSync.Full)
    transpose_faces_modes = (Transpose.No, Transpose.Yes)

    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
                for dest_sync in dest_sync_modes:
                    for math_transpose_faces in transpose_faces_modes:
                        for dimensions in dimensions_cache[(dest_acc, dest_sync)]:
                            combinations.append(
                                (
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    math_transpose_faces,
                                    dimensions,
                                )
                            )

    return combinations


TRANSPOSE_DEST_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Int8,
        DataFormat.UInt8,
    ],
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_transpose_dims=generate_qsr_transpose_dest_combinations(
        TRANSPOSE_DEST_FORMATS
    ),
    implied_math_format=[ImpliedMathFormat.No],
)
def test_transpose_dest_quasar(
    formats_dest_acc_sync_transpose_dims,
    implied_math_format,
):
    (formats, dest_acc, dest_sync, math_transpose_faces, input_dimensions) = (
        formats_dest_acc_sync_transpose_dims
    )

    data_copy_type = DataCopyType.A2D
    num_faces = 4

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Generate custom test input stimuli to check large Int32 and Float32 values
    if (
        formats.input_format == DataFormat.Int32
        and formats.output_format == DataFormat.Int32
    ):
        lo, hi = -1_000_000, 1_000_000
        n = src_A.numel()
        src_A = torch.randint(lo, hi, (n,), dtype=torch.int32).reshape_as(src_A)
        src_B = torch.randint(lo, hi, (n,), dtype=torch.int32).reshape_as(src_B)

    if formats.input_format == DataFormat.Float32:
        n = src_A.numel()
        src_A = (torch.randn(n, dtype=torch.float32) * 10000.0).reshape_as(src_A)
        src_B = (torch.randn(n, dtype=torch.float32) * 10000.0).reshape_as(src_B)

    generate_datacopy_golden = get_golden_generator(DataCopyGolden)
    datacopy_tensor = generate_datacopy_golden(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
    )

    t_matrix = get_golden_generator(TransposeGolden)
    golden_tensor = t_matrix.transpose_within_faces_multi_tile(
        datacopy_tensor,
        formats.output_format,
        num_tiles=tile_cnt_A,
        untilize=False,
        input_dimensions=input_dimensions,
    )
    if math_transpose_faces == Transpose.Yes:
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )

    unpack_to_dest = (
        True
        if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        else False
    )
    configuration = TestConfig(
        "sources/quasar/transpose_dest_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(data_copy_type),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(),
        ],
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
