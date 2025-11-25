# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
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
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, run_test
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
    test_name="eltwise_binary_test",
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
    test_name,
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

    # Generate stimuli for both operands
    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
    )

    # Generate golden result using eltwise binary golden generator
    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
    )

    # Determine unpack_to_dest based on format and accumulation mode
    # This follows the same logic as pack_test
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "mathop": mathop,
        "math_fidelity": math_fidelity,
        "implied_math_format": implied_math_format,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
        "num_faces": num_faces,
    }

    # Write both operands to L1 memory
    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
        num_faces=num_faces,
    )

    # Run the C++ kernel
    run_test(test_config, boot_mode=boot_mode)

    # Collect results from L1 memory
    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt, address=res_address, num_faces=num_faces
    )

    # Verify results match golden
    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
