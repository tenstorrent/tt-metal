# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    DataCopyGolden,
    get_golden_generator,
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
    calculate_edgecase_dest_indices,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import SUPPORTED_TILE_SIZES, is_mx_unsupported_tile_dims
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test


def generate_eltwise_unary_datacopy_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate eltwise_unary_datacopy combinations.

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, data_copy_type, input_dimensions, dest_sync, edgecase_dest_index, tile_dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (DestAccumulation.No, DestAccumulation.Yes)
        dest_sync_modes = (DestSync.Half, DestSync.Full)
        data_copy_types = (DataCopyType.A2D, DataCopyType.B2D)

        for dest_acc in dest_acc_modes:
            if (
                in_fmt != DataFormat.Float32
                and fmt.output_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue

            for dest_sync in dest_sync_modes:
                for data_copy_type in data_copy_types:
                    for tile_dims in SUPPORTED_TILE_SIZES:
                        if is_mx_unsupported_tile_dims(
                            in_fmt, fmt.output_format, tile_dims
                        ):
                            continue
                        tile_shape = construct_tile_shape(tile_dims)
                        for dimensions in generate_unary_input_dimensions(
                            dest_acc, dest_sync=dest_sync, tile_shape=tile_shape
                        ):
                            for (
                                _,
                                edgecase_dest_index,
                            ) in calculate_edgecase_dest_indices(
                                True if dest_acc == DestAccumulation.Yes else False,
                                dimensions[0]
                                // tile_dims[0]
                                * dimensions[1]
                                // tile_dims[1],
                                [dest_sync],
                            ):
                                combinations.append(
                                    (
                                        fmt,
                                        dest_acc,
                                        data_copy_type,
                                        runtime(dimensions),
                                        dest_sync,
                                        runtime(edgecase_dest_index),
                                        runtime(tile_dims),
                                    )
                                )

    return combinations


DATACOPY_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ]
)
ALL_DATACOPY_COMBINATIONS = generate_eltwise_unary_datacopy_combinations(
    DATACOPY_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices=ALL_DATACOPY_COMBINATIONS,
    # don't generate the No variant for them. combo[0] is the InputOutputFormat (input/output pair).
    implied_math_format=lambda formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices: (
        [ImpliedMathFormat.Yes]
        if formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices[
            0
        ].input_format.is_mx_format()
        else [ImpliedMathFormat.Yes, ImpliedMathFormat.No]
    ),
)
def test_eltwise_unary_datacopy_quasar(
    formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices,
    implied_math_format,
):
    (
        formats,
        dest_acc,
        data_copy_type,
        input_dimensions,
        dest_sync_mode,
        dest_index,
        tile_dimensions,
    ) = formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices

    # MX formats REQUIRE implied_math_format=Yes on Quasar (bypass format inference pipeline)
    if (
        formats.input_format.is_mx_format()
        and implied_math_format == ImpliedMathFormat.No
    ):
        pytest.skip("MX formats require implied_math_format=Yes on Quasar")

    tile_shape = construct_tile_shape(tile_dimensions)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_faces = tile_shape.total_num_faces()

    golden_src = src_B if data_copy_type == DataCopyType.B2D else src_A
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        golden_src,
        formats.output_format,
        num_faces=num_faces,
        face_r_dim=tile_shape.face_r_dim,
        input_dimensions=input_dimensions,
        input_format=formats.input_format,
        tile_shape=tile_shape,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_unary_datacopy_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(data_copy_type),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpB
                if data_copy_type == DataCopyType.B2D
                else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            DEST_INDEX(dest_index),
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting
        disable_format_inference=(
            implied_math_format == ImpliedMathFormat.Yes
            and formats.input_format.is_mx_format()
        ),
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
    ), "Assert against golden failed"
