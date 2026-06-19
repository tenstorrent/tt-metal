# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

# Embedded piecewise-polynomial LUT — MUST match the C++ kernel
# (sources/generic_lut_activation_custom_test.cpp) exactly.
#   NUM_SEGMENTS = 4, POLY_DEGREE = 2, sigmoid approximation on [-4, 4].
#   LUT layout: [b0..b4, then per segment (POLY_DEGREE+1) coeffs c0,c1,c2].
NUM_SEGMENTS = 4
POLY_DEGREE = 2
BOUNDARIES = [-4.0, -2.0, 0.0, 2.0, 4.0]
COEFFS = [
    [0.38296354, 0.17515847, 0.02109685],  # seg0: c0, c1, c2
    [0.50329190, 0.27505103, 0.04113654],  # seg1
    [0.49670810, 0.27505103, -0.04113654],  # seg2
    [0.61703646, 0.17515847, -0.02109685],  # seg3
]


def _eval_poly_deg2(coeffs, x):
    # Horner: (c2*x + c1)*x + c0
    return (coeffs[2] * x + coeffs[1]) * x + coeffs[0]


def piecewise_generic_lut_golden(x: torch.Tensor) -> torch.Tensor:
    """Replicate the EXACT kernel algorithm in float64:
    clamp x to [b0, bN], select segment by boundaries on the clamped value,
    Horner-eval that segment's coeffs."""
    x = x.to(torch.float64)
    x_clamped = torch.clamp(x, BOUNDARIES[0], BOUNDARIES[NUM_SEGMENTS])

    # Start with segment 0, then override with the correct segment.
    result = _eval_poly_deg2(COEFFS[0], x_clamped)
    for seg in range(1, NUM_SEGMENTS):
        mask = x_clamped >= BOUNDARIES[seg]
        result = torch.where(mask, _eval_poly_deg2(COEFFS[seg], x_clamped), result)
    return result


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
)
def test_generic_lut_activation_custom(
    formats,
    dest_acc,
):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    num_faces = 4
    tile_cnt = 1

    # Input within the LUT boundaries [-4, 4], as bfloat16 (Float16_b).
    num_elements = input_dimensions[0] * input_dimensions[1]
    src_A = (torch.rand(num_elements, dtype=torch.float32) * 8.0 - 4.0).to(
        torch.bfloat16
    )
    src_B = torch.zeros(num_elements, dtype=torch.bfloat16)
    tile_cnt_A = tile_cnt
    tile_cnt_B = tile_cnt

    # Golden: apply the piecewise LUT to the (bf16-quantized) input.
    golden_values = piecewise_generic_lut_golden(src_A.to(torch.float32))
    golden_tensor = golden_values.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/generic_lut_activation_custom_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(num_faces)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    outcome = configuration.run()
    res_from_L1 = outcome.result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        print_pcc=True,
    )
