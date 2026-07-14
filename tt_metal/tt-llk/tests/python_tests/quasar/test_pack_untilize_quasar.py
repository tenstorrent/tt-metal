# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
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
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
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

PACK_UNTILIZE_TILE_SIZES = [
    (32, 32),
    # (16, 32),
    (1, 32),
    (2, 32),
    # (4, 32),
]


def generate_pack_untilize_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack_untilize combinations.

    Args:
        formats_list: List of input-output format pairs

    Returns: List of (format, dest_acc, dest_sync, input_dimensions, tile_dimensions) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        # If input format is Int16, output format must also be Int16, and vice versa
        if (in_fmt == DataFormat.Int16) ^ (out_fmt == DataFormat.Int16):
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        # Int16 requires 16bit mode dest register
        if in_fmt == DataFormat.Int16:
            return (DestAccumulation.No,)
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    dest_sync_modes = (DestSync.Half, DestSync.Full)
    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        # MX as output format produces flaky results on Quasar.
        if out_fmt.is_mx_format():
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            for dest_sync in dest_sync_modes:
                for tile_dims in PACK_UNTILIZE_TILE_SIZES:
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
                        # if dimensions != list(tile_dims):
                        #     continue
                        # if dimensions[1] != 32:
                        #     continue
                        if list(dimensions) != [32, 32]:
                            continue
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                dest_sync,
                                runtime(dimensions),
                                runtime(tile_dims),
                            )
                        )

    return combinations


PACK_UNTILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Int16,
        DataFormat.Int32,
        # DataFormat.MxFp4,
        # DataFormat.MxInt8,
        # DataFormat.MxInt4,
        # DataFormat.MxInt2,
    ],
)
ALL_PACK_UNTILIZE_COMBINATIONS = generate_pack_untilize_combinations(
    PACK_UNTILIZE_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_dimensions_tile_dims=ALL_PACK_UNTILIZE_COMBINATIONS,
)
def test_pack_untilize_quasar(formats_dest_acc_sync_dimensions_tile_dims):
    (formats, dest_acc, dest_sync_mode, input_dimensions, tile_dimensions) = (
        formats_dest_acc_sync_dimensions_tile_dims[0]
    )

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

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
        tile_dimensions=tile_dimensions,
    )

    num_faces = tile_shape.total_num_faces()
    configuration = TestConfig(
        "sources/quasar/pack_untilize_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=tile_dimensions
            ),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
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
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting.
        disable_format_inference=(
            formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
        ),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
