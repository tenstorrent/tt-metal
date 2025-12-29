# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
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
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test

# Quasar hardware constraints for eltwise operations
TILE_DIM = 32  # Standard tile dimension (32x32)
MAX_TILES_16_BIT_DEST = 8  # Max tiles with 16-bit dest (Float16/Float16_b)

ELTWISE_DIMENSIONS = [
    ([mt_dim * TILE_DIM, nt_dim * TILE_DIM], DestAccumulation.No)
    for mt_dim in range(1, MAX_TILES_16_BIT_DEST + 1)
    for nt_dim in range(1, MAX_TILES_16_BIT_DEST // mt_dim + 1)
]


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    implied_math_format=[
        ImpliedMathFormat.No,
        ImpliedMathFormat.Yes,
    ],
    dimensions_dest_acc=ELTWISE_DIMENSIONS,
    num_faces=[4],
)
def test_eltwise_binary(
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dimensions_dest_acc,
    num_faces,
    boot_mode=BootMode.DEFAULT,
):

    # Unpack dimensions and dest_acc from the tuple
    input_dimensions, dest_acc = dimensions_dest_acc

    # Math fidelity only affects multiplication operations
    if (
        mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        and math_fidelity != MathFidelity.LoFi
    ):
        pytest.skip("Math fidelity only affects multiplication operations")

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
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
        # Determine unpack_to_dest based on format and accumulation mode
        # This follows the same logic as pack_test
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run()

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
