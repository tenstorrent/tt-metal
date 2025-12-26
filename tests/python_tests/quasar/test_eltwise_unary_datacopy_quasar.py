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
    ImpliedMathFormat,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    calculate_edgecase_dest_indices,
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
    INPUT_DIMENSIONS,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def generate_eltwise_unary_datacopy_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate eltwise_unary_datacopy combinations.

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, data_copy_type, input_dimensions, edgecase_dest_index) tuples
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

        dest_acc_modes = (DestAccumulation.No, DestAccumulation.Yes)
        data_copy_types = (DataCopyType.A2D, DataCopyType.B2D)

        for dest_acc in dest_acc_modes:
            if (
                in_fmt != DataFormat.Float32
                and fmt.output_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                # Skip if input format is not Float32 and output format is Float32 and dest_acc is No
                # This combination is not supported in the Quasar Packer format conversions
                continue

            for data_copy_type in data_copy_types:
                for dimensions in dimensions_cache[dest_acc]:
                    for _, edgecase_dest_index in calculate_edgecase_dest_indices(
                        True if dest_acc == DestAccumulation.Yes else False,
                        dimensions[0] // 32 * dimensions[1] // 32,
                    ):
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                data_copy_type,
                                dimensions,
                                edgecase_dest_index,
                            )
                        )

    return combinations


DATACOPY_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ]
)
ALL_DATACOPY_COMBINATIONS = generate_eltwise_unary_datacopy_combinations(
    DATACOPY_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_data_copy_type_dims_dest_indices=ALL_DATACOPY_COMBINATIONS,
    implied_math_format=[ImpliedMathFormat.Yes, ImpliedMathFormat.No],
)
def test_eltwise_unary_datacopy_quasar(
    formats_dest_acc_data_copy_type_dims_dest_indices,
    implied_math_format,
):
    formats = formats_dest_acc_data_copy_type_dims_dest_indices[0]
    dest_acc = formats_dest_acc_data_copy_type_dims_dest_indices[1]
    data_copy_type = formats_dest_acc_data_copy_type_dims_dest_indices[2]
    input_dimensions = formats_dest_acc_data_copy_type_dims_dest_indices[3]
    dest_index = formats_dest_acc_data_copy_type_dims_dest_indices[4]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    num_faces = 4

    golden_src = src_B if data_copy_type == DataCopyType.B2D else src_A
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        golden_src,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_unary_datacopy_quasar_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(data_copy_type),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpB
                if data_copy_type == DataCopyType.B2D
                else UnpackerEngine.UnpA
            ),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_SYNC(),
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX(dest_index),
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
    )

    res_from_L1 = configuration.run()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
