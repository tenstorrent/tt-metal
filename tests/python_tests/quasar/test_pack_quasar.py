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
from helpers.llk_params import DestAccumulation, ImpliedMathFormat, format_dict
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
    INPUT_DIMENSIONS,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test


def generate_qsr_pack_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, input_dimensions) tuples
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
        for dest_acc in dest_acc_modes:
            if (
                in_fmt != DataFormat.Float32
                and fmt.output_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                # Skip if input format is not Float32 and output format is Float32 and dest_acc is No
                # This combination is not supported in the Quasar Packer format conversions
                continue
            for dimensions in dimensions_cache[dest_acc]:
                combinations.append((fmt, dest_acc, dimensions))

    return combinations


PACK_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_input_dims=generate_qsr_pack_combinations(PACK_FORMATS),
)
def test_pack_quasar(formats_dest_acc_input_dims, boot_mode=BootMode.DEFAULT):
    (formats, dest_acc, input_dimensions) = formats_dest_acc_input_dims[0]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    num_faces = 4
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
    )

    configuration = TestConfig(
        "sources/quasar/pack_quasar_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(),
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
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
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
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
