# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_hypot(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -101, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -52, 51, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device, True)

    tt_output_tensor_on_device = ttnn.hypot_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hypot_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "scale_exp, atol",
    [
        (80, 1e-2),
        (-80, 1e-2),
        (0, 1e-5),
    ],
    ids=["large_exp_2p80", "small_exp_2m80", "zero_inputs"],
)
def test_bw_hypot_extreme_and_zero(scale_exp, atol, device):
    shape = torch.Size([1, 1, 32, 32])
    torch.manual_seed(0)
    if scale_exp == 0:
        in_data = torch.zeros(shape, dtype=torch.bfloat16)
        other_data = torch.zeros(shape, dtype=torch.bfloat16)
    else:
        scale = math.ldexp(1.0, scale_exp)
        in_data = (
            torch.empty(shape, dtype=torch.float32).uniform_(0.5, 2.0) * scale
        ).to(torch.bfloat16)
        other_data = (
            torch.empty(shape, dtype=torch.float32).uniform_(-2.0, -0.5) * scale
        ).to(torch.bfloat16)
    grad_data = torch.ones(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    other_tensor = ttnn.from_torch(
        other_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    grad_tensor = ttnn.from_torch(
        grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = ttnn.hypot_bw(grad_tensor, input_tensor, other_tensor)
    grad_a = ttnn.to_torch(tt_output[0]).to(torch.float32)
    grad_b = ttnn.to_torch(tt_output[1]).to(torch.float32)

    assert not torch.isnan(grad_a).any() and not torch.isnan(grad_b).any()
    assert not torch.isinf(grad_a).any() and not torch.isinf(grad_b).any()

    if scale_exp == 0:
        golden_a = torch.zeros_like(grad_a)
        golden_b = torch.zeros_like(grad_b)
    else:
        in_f64 = in_data.to(torch.float64).requires_grad_(True)
        other_f64 = other_data.to(torch.float64).requires_grad_(True)
        torch.hypot(in_f64, other_f64).backward(grad_data.to(torch.float64))
        golden_a = in_f64.grad.to(torch.float32)
        golden_b = other_f64.grad.to(torch.float32)

    assert torch.allclose(grad_a, golden_a, rtol=2e-2, atol=atol)
    assert torch.allclose(grad_b, golden_b, rtol=2e-2, atol=atol)
