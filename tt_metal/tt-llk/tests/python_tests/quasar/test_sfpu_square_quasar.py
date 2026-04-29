# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_sfpu_format_dest_acc_combinations,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator_v2 import generate_stimuli_v2
from helpers.stimuli_generator import (
    apply_log_uniform_magnitudes,
    compute_safe_input_magnitude_range,
    format_elem_max,
)
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

# Safety factor applied to format maxima when clamping square operands.
# Keeps x² comfortably within the output format and |x| within the input format.
SQUARE_RANGE_SAFETY_FACTOR = 0.9

SFPU_SQUARE_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)

SFPU_SQUARE_COMBINATIONS = [
    (fmt, dest_acc, dest_sync, implied_math_format, input_dimensions)
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(SFPU_SQUARE_FORMATS)
    for dest_sync in (DestSync.Half, DestSync.Full)
    for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    for input_dimensions in [[32, 32], [64, 64]]
]


@pytest.mark.quasar
@parametrize(formats_dest_acc_sync_implied_math_input_dims=SFPU_SQUARE_COMBINATIONS)
def test_sfpu_square_quasar(formats_dest_acc_sync_implied_math_input_dims):
    """
    Test square operation on Quasar architecture.

    Uses PyTorch's square (x**2) as the golden reference and generates input stimuli
    covering the full representable range
    """
    (formats, dest_acc, dest_sync, implied_math_format, input_dimensions) = (
        formats_dest_acc_sync_implied_math_input_dims[0]
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli_v2(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Both caps invert the squaring op so x² stays representable. For the input
    # cap this is because the SFPU squares in the input format's math precision
    # (except for MX, where squaring uses a wider intermediate and |x| itself
    # is the binding constraint -- mx_elem_max is already small).
    input_elem_max = format_elem_max(formats.input_format)
    input_magnitude_cap = (
        input_elem_max
        if formats.input_format.is_mx_format()
        else math.sqrt(input_elem_max)
    ) * SQUARE_RANGE_SAFETY_FACTOR
    output_magnitude_cap = (
        math.sqrt(format_elem_max(formats.output_format)) * SQUARE_RANGE_SAFETY_FACTOR
    )

    min_magnitude, max_magnitude = compute_safe_input_magnitude_range(
        formats.input_format,
        formats.output_format,
        input_magnitude_cap=input_magnitude_cap,
        output_magnitude_cap=output_magnitude_cap,
    )
    src_A = apply_log_uniform_magnitudes(
        src_A,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        cast_to_format=formats.input_format,
        sign_source=src_B,
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Square,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        "sources/quasar/sfpu_square_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Square),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
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
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
