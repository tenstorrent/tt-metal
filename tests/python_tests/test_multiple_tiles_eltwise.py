# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
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
    INPUT_DIMENSIONS,
    MATH_FIDELITY,
    MATH_OP,
    TILE_COUNT,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Bfp8_b,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ]
    ),
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    input_dimensions=[[32, 32], [32, 64], [64, 64]],
)
def test_multiple_tiles(
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    input_dimensions,
    workers_tensix_coordinates,
):

    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop, src_A, src_B, formats.output_format, math_fidelity
    )

    configuration = TestConfig(
        "sources/multiple_tiles_eltwise_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
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
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
