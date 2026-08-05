# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    (
        0.0,
        0.5,
        1.5,
        2.0,
        3.0,
        5.7,
        15.2,
    ),
)
def test_bw_rpow(input_shapes, exponent, device):
    # rpow(input, exponent) is exponent ** input, so the input range has to keep
    # exponent ** input representable. The previous range of (-201, 199) saturated to
    # inf or 0 for every exponent above ~1.6, which made the comparison meaningless.
    # With (-20, 20) the largest value here is 15.2 ** 20 = 4.2e23, well inside bfloat16.
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -20, 20, device, True)

    tt_output_tensor_on_device = ttnn.rpow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.rpow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("exponent", (2.0, 3.0, 10.0))
@pytest.mark.parametrize("input_value", (-5.0, -0.5, -20.0))
def test_bw_rpow_negative_input_has_a_finite_gradient(exponent, input_value, device):
    # rpow(-5, 2) is 2 ** -5 = 0.03125, an ordinary number, so its derivative is ordinary
    # too: ln(2) * 2 ** -5 = 0.02166. Negative inputs used to be overwritten with NaN.
    #
    # compare_pcc cannot catch this on its own: get_pcc zeroes NaN and Inf on both sides
    # before correlating, so this test compares the values directly.
    shape = torch.Size([1, 1, 32, 32])
    torch_input = torch.full(shape, input_value, dtype=torch.float32)
    torch_grad = torch.ones(shape, dtype=torch.float32)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = ttnn.to_torch(ttnn.rpow_bw(grad_tensor, input_tensor, exponent)[0]).float()

    assert torch.isfinite(tt_out).all(), (
        f"rpow_bw({exponent}) returned a non-finite gradient at input {input_value}, "
        f"where the forward value {exponent} ** {input_value} is finite"
    )

    expected = math.log(exponent) * exponent**input_value
    assert torch.allclose(tt_out, torch.full_like(tt_out, expected), rtol=0.05, atol=0.0), (
        f"rpow_bw({exponent}) at input {input_value}: got {tt_out.flatten()[0].item()}, expected {expected}"
    )


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
def test_bw_rpow_base_two_agrees_with_exp2_bw(input_shapes, device):
    # ttnn.exp2(x) and ttnn.rpow(x, 2.0) compute the same function, so their backward
    # passes have to agree. exp2_bw already implements grad * ln(2) * 2 ** x.
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -20, 20, device, True)

    rpow_out = ttnn.to_torch(ttnn.rpow_bw(grad_tensor, input_tensor, 2.0)[0]).float()
    exp2_out = ttnn.to_torch(ttnn.exp2_bw(grad_tensor, input_tensor)[0]).float()

    assert torch.allclose(rpow_out, exp2_out, rtol=0.05, atol=1e-6), (
        "rpow_bw(x, 2.0) and exp2_bw(x) differ, but exp2(x) and rpow(x, 2.0) are the same function"
    )
