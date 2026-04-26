# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES
from helpers.tile_shape import construct_tile_shape
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
            for dest_sync in (DestSync.Half, DestSync.Full):
                for transpose_en in transpose_modes:
                    for unpacker_sel in unpacker_engines:
                        # transpose is not supported for tiny-tiles
                        tile_sizes = (
                            ((32, 32),)
                            if transpose_en == Transpose.Yes
                            else SUPPORTED_TILE_SIZES
                        )
                        for tile_dims in tile_sizes:
                            tile_shape = construct_tile_shape(tile_dims)
                            for dimensions in generate_unary_input_dimensions(
                                dest_acc, dest_sync=dest_sync, tile_shape=tile_shape
                            ):
                                combinations.append(
                                    (
                                        fmt,
                                        dest_acc,
                                        dest_sync,
                                        transpose_en,
                                        unpacker_sel,
                                        dimensions,
                                        tile_dims,
                                    )
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
    formats_dest_acc_sync_transpose_unpack_sel_dims=ALL_UNPACK_UNARY_OPERAND_COMBINATIONS,
)
def test_unpack_unary_operand_quasar(
    formats_dest_acc_sync_transpose_unpack_sel_dims,
    boot_mode=BootMode.DEFAULT,
):
    (
        formats,
        dest_acc,
        dest_sync_mode,
        transpose_en,
        unpacker_sel,
        input_dimensions,
        tile_dimensions,
    ) = formats_dest_acc_sync_transpose_unpack_sel_dims[0]

    tile_shape = construct_tile_shape(tile_dimensions)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_faces = tile_shape.total_num_faces()

    golden_src = (
        src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    )  # use A for UnpA and UnpDest
    if transpose_en == Transpose.Yes:
        generate_golden = get_golden_generator(TransposeGolden)
        golden_tensor = generate_golden.transpose_faces_multi_tile(
            golden_src,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden_tensor = generate_golden.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    else:
        generate_golden = get_golden_generator(DataCopyGolden)
        golden_tensor = generate_golden(
            golden_src,
            formats.output_format,
            num_faces=num_faces,
            input_dimensions=input_dimensions,
            face_r_dim=tile_shape.face_r_dim,
            tile_shape=tile_shape,
        )

    configuration = TestConfig(
        "sources/quasar/unpack_unary_operand_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(transpose_en),
            UNPACK_TRANS_WITHIN_FACE(transpose_en),
        ],
        runtimes=[
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
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
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
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
        golden_tensor, res_tensor, formats.output_format, tile_shape=tile_shape
    ), "Assert against golden failed"
