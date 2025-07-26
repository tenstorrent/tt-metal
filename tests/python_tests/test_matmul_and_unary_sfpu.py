# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    ApproximationMode,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@parametrize(
    test_name="matmul_and_unary_sfpu_test",
    formats=input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Bfp8_b,
        ],
        same=True,
    ),
    mathop=[
        MathOperation.Abs,
        MathOperation.Celu,
        MathOperation.Cos,
        MathOperation.Gelu,
        MathOperation.Hardsigmoid,
        MathOperation.Log,
        MathOperation.Reciprocal,
        MathOperation.Silu,
        MathOperation.Sin,
        MathOperation.Sqrt,
        MathOperation.Square,
    ],
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        # MathFidelity.HiFi2, TODO: FIND OUT WHY
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_matmul_and_unary_sfpu(
    test_name, formats, mathop, approx_mode, dest_acc, math_fidelity
):
    input_dimensions = [32, 32]

    if mathop in [MathOperation.Cos, MathOperation.Sin]:
        pytest.skip("Cos and Sin operations are not fully functional yet")
    if mathop == MathOperation.Square and math_fidelity == MathFidelity.LoFi:
        pytest.skip("Square operation in LoFi is not fully functional yet")
    if (
        formats.input_format == formats.output_format == DataFormat.Float16
        and mathop
        in [
            MathOperation.Log,
            MathOperation.Sqrt,
            MathOperation.Square,
            MathOperation.Hardsigmoid,
        ]
        and dest_acc == DestAccumulation.No
        and get_chip_architecture() == ChipArchitecture.BLACKHOLE
    ):
        pytest.skip("BFP8 does not support Log and Reciprocal operations")

    torch_format = format_dict.get(formats.output_format)
    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_matmul_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_matmul_golden(
        src_A, src_B, formats.output_format, math_fidelity
    )
    golden_tensor = tilize(golden_tensor, formats.output_format)

    generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_sfpu_golden(
        mathop, golden_tensor, formats.output_format, dest_acc, formats.input_format
    )
    golden_tensor = golden_tensor.to(torch_format)

    res_address = write_stimuli_to_l1(
        tilize(src_A, formats.input_format),
        tilize(src_B, formats.input_format),
        formats.input_format,
        formats.input_format,
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "approx_mode": approx_mode,
        "mathop": mathop,
        "L1_to_L1_iterations": 2,  # This is a fused test does two runs of L1-L1, result tensor from first run (matmul) is used as input for second run (sfpu operation)
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        test_config.get(
            "L1_to_L1_iterations"  # Needed to calculate accumulated precision loss for fused tests that copy result tensor as input for next runs
        ),
    )
