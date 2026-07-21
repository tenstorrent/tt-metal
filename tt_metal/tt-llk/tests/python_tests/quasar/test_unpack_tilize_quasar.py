# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
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
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tile_constants import (
    MX_SUPPORTED_TILE_SIZES,
    is_mx_unsupported_tile_dims,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

UNPACK_TILIZE_TILE_SIZES = [
    (32, 32),
    (1, 32),
    (2, 32),
]


def generate_unpack_tilize_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate unpack_tilize combinations.

    Rules:
    1. 32-bit formats require DestAccumulation.Yes

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, dest_sync, unpacker_sel, input_dimensions,
             tile_dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format
        out_fmt = fmt.output_format

        dest_acc_modes = (
            # disable dest accumulation for now
            # (DestAccumulation.Yes,)
            # if in_fmt.is_32_bit()
            # else (
            #     (DestAccumulation.No,)
            #     if in_fmt in [DataFormat.Float16, DataFormat.Int16]
            #     else (DestAccumulation.No, DestAccumulation.Yes)
            # )
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No,)
        )
        # 32-bit tilize uses unpack_to_dest (UNP_DEST)
        unpacker_engines = (
            (UnpackerEngine.UnpDest,)
            if in_fmt.is_32_bit()
            else (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
        )

        for dest_acc in dest_acc_modes:
            for dest_sync in (DestSync.Half, DestSync.Full):
                for unpacker_sel in unpacker_engines:
                    for tile_dims in UNPACK_TILIZE_TILE_SIZES:
                        if is_mx_unsupported_tile_dims(in_fmt, out_fmt, tile_dims):
                            continue
                        if (
                            in_fmt.is_32_bit()
                            and dest_acc == DestAccumulation.Yes
                            and tile_dims not in MX_SUPPORTED_TILE_SIZES
                        ):
                            continue
                        tile_shape = construct_tile_shape(tile_dims)
                        for dimensions in generate_unary_input_dimensions(
                            dest_acc, dest_sync=dest_sync, tile_shape=tile_shape
                        ):
                            combinations.append(
                                (
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    unpacker_sel,
                                    runtime(dimensions),
                                    runtime(tile_dims),
                                )
                            )

    return combinations


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,  # Input format and output format are the same
)
ALL_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_unpack_sel_dimensions_tile_dims=ALL_UNPACK_TILIZE_COMBINATIONS,
)
def test_unpack_tilize_quasar(
    formats_dest_acc_sync_unpack_sel_dimensions_tile_dims, boot_mode=BootMode.DEFAULT
):
    (
        formats,
        dest_acc,
        dest_sync_mode,
        unpacker_sel,
        input_dimensions,
        tile_dimensions,
    ) = formats_dest_acc_sync_unpack_sel_dimensions_tile_dims[0]

    tile_shape = construct_tile_shape(tile_dimensions)

    sequential_spec = StimuliSpec.sequential()
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=sequential_spec,
        spec_B=sequential_spec,
    )

    num_faces = tile_shape.total_num_faces()

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    if formats.input_format.is_mx_format():
        golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)
    golden_tensor = generate_golden(
        golden_src,
        input_dimensions,
        formats.output_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )

    configuration = TestConfig(
        "sources/quasar/unpack_tilize_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
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
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting.
        disable_format_inference=(formats.input_format.is_mx_format()),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
