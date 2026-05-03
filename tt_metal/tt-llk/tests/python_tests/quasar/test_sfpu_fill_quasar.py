# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
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
    FILL_INT_FORMAT,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def generate_sfpu_fill_combinations(
    float_formats: List[FormatConfig],
    int_formats: List[FormatConfig],
):
    """
    Generate SFPU fill test combinations.

    fill is a universal op: it writes a constant to every element of Dest,
    independent of the current values. The format matrix covers float and
    integer formats; the kernel path (and FILL_INT_FORMAT template) is
    chosen by the test based on whether the format is integer.

    Args: Input-output format pairs for float and int paths

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions) tuples
    """
    combinations = []

    for fmt in float_formats:
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

    # Int fill: _calculate_fill_int_ with FILL_INT_FORMAT-selected SFPMEM store mode.
    # Int32 is 32-bit → dest_acc Yes; Int16 (16-bit) and Int8/UInt8 (8-bit) → dest_acc No.
    # Only ImpliedMathFormat.No is exercised on the int path.
    for fmt in int_formats:
        in_fmt = fmt.input_format
        dest_acc = DestAccumulation.Yes if in_fmt.is_32_bit() else DestAccumulation.No
        for input_dimensions in [[32, 32], [64, 64]]:
            combinations.append((fmt, dest_acc, ImpliedMathFormat.No, input_dimensions))

    return combinations


_FILL_FLOAT_INPUTS = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
]
_FILL_FLOAT_OUTPUTS = _FILL_FLOAT_INPUTS + [
    DataFormat.MxFp8R,
    DataFormat.MxFp8P,
]
SFPU_FILL_FLOAT_FORMATS = [
    InputOutputFormat(in_fmt, out_fmt)
    for in_fmt in _FILL_FLOAT_INPUTS
    for out_fmt in _FILL_FLOAT_OUTPUTS
]

# Fill int path: _calculate_fill_int_ with p_sfpu::sfpmem::INT32/UINT16/UINT8.
# Quasar integer formats and their SFPMEM store modes:
#   Int32  → sfpmem::INT32  (32-bit sign-magnitude)
#   Int16  → sfpmem::UINT16 (INT16 = Quasar hardware code 9, maps to SFPMEM::UINT16)
#   Int8   → sfpmem::UINT8  (8-bit)
#   UInt8  → sfpmem::UINT8  (8-bit unsigned)
# Note: UInt16 (Quasar code 130) is invalid on Quasar; Int16 (code 9) is the correct
# 16-bit integer format. FILL_INT_FORMAT bakes the format into each compiled variant.
SFPU_FILL_INT_FORMATS = input_output_formats(
    [DataFormat.Int32, DataFormat.Int16, DataFormat.Int8, DataFormat.UInt8],
    same=True,
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims=generate_sfpu_fill_combinations(
        SFPU_FILL_FLOAT_FORMATS, SFPU_FILL_INT_FORMATS
    ),
)
def test_sfpu_fill_quasar(formats_dest_acc_implied_math_input_dims):
    """
    Test fill operation on Quasar architecture.

    fill writes a scalar constant to every element of a Dest tile.

    Float formats use _calculate_fill_ (SFPU DEFAULT store mode) with
    FILL_CONST_VALUE = 5.0; integer formats use _calculate_fill_int_ with
    FILL_INT_FORMAT baked in as a constexpr so the SFPMEM store mode is
    selected at compile time with no runtime dispatch (kernel writes
    FILL_INT_VALUE = 5 via SFPLOADI + SFPSTORE). The merged kernel selects
    the int vs float path at runtime from formats.unpack_A_src; FILL_INT_FORMAT
    is always forwarded with a value that is safe for _calculate_fill_int_'s
    static_assert (the input format on the int path, Int32 placeholder on the
    float path).

    Since fill ignores input values, the input stimuli are arbitrary —
    typed stimuli are still generated so the unpack path sees a valid buffer.
    """
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )

    is_int_fill = formats.input_format.is_integer()

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

    if is_int_fill:
        # FILL_INT_VALUE must match FILL_INT_VALUE in sfpu_fill_quasar_test.cpp
        FILL_INT_VALUE = 5
        num_elements = src_A.numel()
        golden_tensor = torch.full(
            (num_elements,),
            FILL_INT_VALUE,
            dtype=format_dict[formats.output_format],
        )
    else:
        # FILL_CONST_VALUE must match FILL_CONST = 5.0f in sfpu_fill_quasar_test.cpp
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
            FILL_INT_FORMAT(formats.input_format if is_int_fill else DataFormat.Int32),
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
