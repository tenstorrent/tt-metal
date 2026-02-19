# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from conftest import skip_for_blackhole, skip_for_coverage, skip_for_wormhole
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    MatmulGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


# SFPI Issue link:
# When some of these SPFU ops get compiled with coverage, `#pragma GCC unroll X` marked loops become invalid assembly
@skip_for_coverage
@skip_for_blackhole
@skip_for_wormhole
@parametrize(
    test_name="sources/matmul_and_unary_sfpu_test.cpp",
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
        # MathOperation.Gelu,
        MathOperation.Hardsigmoid,
        MathOperation.Log,
        MathOperation.Reciprocal,
        # MathOperation.Silu,
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
    test_name,
    formats,
    mathop,
    approx_mode,
    dest_acc,
    math_fidelity,
    workers_tensix_coordinates,
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
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_matmul_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_matmul_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_dimensions,
        input_B_dimensions=input_dimensions,
    )
    golden_tensor = tilize(golden_tensor, formats.output_format)

    generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_sfpu_golden(
        mathop,
        golden_tensor,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )
    golden_tensor = golden_tensor.to(torch_format)

    configuration = TestConfig(
        test_name,
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_FIDELITY(math_fidelity),
            APPROX_MODE(approx_mode),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=2,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format, 2
    ), "Assert against golden failed"
