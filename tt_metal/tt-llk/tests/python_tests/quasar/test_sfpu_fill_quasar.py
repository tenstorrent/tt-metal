# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def generate_sfpu_fill_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU fill test combinations.

    fill is a universal op: it writes a constant to every element of Dest,
    independent of the current values. The format matrix covers float formats.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        # SFPU unpack_to_dest requires bit-width match: 32-bit formats pair with
        # dest_acc=Yes, non-32-bit formats pair with dest_acc=No.
        dest_acc = DestAccumulation.Yes if in_fmt.is_32_bit() else DestAccumulation.No

        # Quasar packer does not support non-Float32 to Float32 conversion when dest_acc=No
        if (
            in_fmt != DataFormat.Float32
            and fmt.output_format == DataFormat.Float32
            and dest_acc == DestAccumulation.No
        ):
            continue

        for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
            for input_dimensions in [[32, 32], [64, 64]]:
                combinations.append(
                    (fmt, dest_acc, implied_math_format, input_dimensions)
                )

    return combinations


# Fill float path: SFPU DEFAULT store mode supports all float formats.
# Int32 requires _calculate_fill_int_ with p_sfpu::sfpmem::INT32 and a separate test.
SFPU_FILL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims=generate_sfpu_fill_combinations(
        SFPU_FILL_FORMATS
    ),
)
def test_sfpu_fill_quasar(formats_dest_acc_implied_math_input_dims):
    """
    Test fill operation on Quasar architecture.

    fill writes a scalar constant (5.0) to every element of a Dest tile.
    The golden reference fills the tensor with the same constant.
    Since fill ignores input values, the input stimuli are arbitrary.
    """
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )

    # Seed for reproducibility
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    num_faces = 4

    # fill_const_value=5.0 matches the FILL_CONST_BITS = 0x40A00000 in the test C++
    FILL_CONST_VALUE = 5.0

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Fill,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
        fill_const_value=FILL_CONST_VALUE,
    )

    # SFPU tests always use unpack_to_dest=True (format matrix pre-filtered to matched bit-widths)
    configuration = TestConfig(
        "sources/quasar/sfpu_fill_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Fill),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
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
        unpack_to_dest=True,
        dest_acc=dest_acc,
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
