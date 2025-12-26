# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    INPUT_DIMENSIONS,
    MATH_FIDELITY,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],  #  Add DataFormat.Bfp8_b only as input when Data format Inference Model 2.0 supports format conversions for > 1 pipeline run with different inputs and outputs.
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
def test_matmul_unpack_tilize(
    formats, dest_acc, math_fidelity, workers_tensix_coordinates
):

    torch_format = format_dict[formats.output_format]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = tilize(
        generate_golden(
            src_A,
            src_B,
            formats.output_format,
            math_fidelity,
            input_A_dimensions=input_dimensions,
            input_B_dimensions=input_dimensions,
        )
    )
    golden_tensor = golden_tensor.to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/matmul_unpack_tilize_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[],
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
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates)

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golder tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        # Needed to calculate accumulated precision loss for fused tests that copy result tensor as input for next runs
        L1_to_L1_iterations=L1_to_L1_iterations,
    ), "Assert against golden failed"
