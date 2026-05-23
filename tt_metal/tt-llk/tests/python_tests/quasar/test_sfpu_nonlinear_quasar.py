# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
from helpers.stimuli_generator import prepare_inputs_for_operation
from helpers.stimuli_generator_v2 import StimuliSpec, generate_stimuli_v2
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

SFPU_NONLINEAR_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)

SFPU_NONLINEAR_MATHOPS = [
    MathOperation.Exp,
    MathOperation.Gelu,
    MathOperation.Relu,
    MathOperation.Reciprocal,
    MathOperation.Sqrt,
    MathOperation.Tanh,
    MathOperation.Sigmoid,
    MathOperation.Silu,
]

SFPU_NONLINEAR_COMBINATIONS = [
    (fmt, dest_acc, dest_sync, implied_math_format, input_dimensions, mathop)
    for fmt, dest_acc in generate_sfpu_format_dest_acc_combinations(
        SFPU_NONLINEAR_FORMATS
    )
    for dest_sync in (DestSync.Half, DestSync.Full)
    for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    for input_dimensions in [[32, 32], [64, 64]]
    for mathop in SFPU_NONLINEAR_MATHOPS
]


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_implied_math_input_dims_mathop=SFPU_NONLINEAR_COMBINATIONS,
)
def test_sfpu_nonlinear_quasar(formats_dest_acc_sync_implied_math_input_dims_mathop):
    """
    Test nonlinear SFPU operations (exp, gelu, relu, reciprocal, sqrt, tanh, sigmoid, silu) on Quasar architecture.

    This test parameterizes over multiple operations to avoid code duplication.
    """
    (formats, dest_acc, dest_sync, implied_math_format, input_dimensions, mathop) = (
        formats_dest_acc_sync_implied_math_input_dims_mathop[0]
    )

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli_v2(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    # Prepare inputs with operation-specific ranges
    src_A = prepare_inputs_for_operation(
        src_A, mathop, formats.input_format, formats.output_format
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
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
        "sources/quasar/sfpu_nonlinear_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
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
