# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [0.05, 0.0, -0.5, 5.12])
def test_bw_unary_div_no_nan(input_shapes, scalar, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 199, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, required_grad=True)

    tt_output_tensor_on_device = ttnn.div_no_nan_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.div_no_nan_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
def test_bw_unary_div_no_nan_subnormal_scalar(dtype, device):
    shape = torch.Size([1, 1, 32, 32])
    scalar = 1.0e-38
    grad_data = torch.linspace(1.0e-37, 1.0e-36, shape.numel(), dtype=torch.float32).reshape(shape)
    in_data = torch.full(shape, 2.0, dtype=torch.float32, requires_grad=True)

    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.div_no_nan_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.div_no_nan_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

    assert compare_pcc(tt_output_tensor_on_device, golden_tensor)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("scalar", [0.0, -0.0], ids=["pos_zero", "neg_zero"])
@pytest.mark.parametrize("grad_value", [float("inf"), float("-inf"), float("nan")], ids=["pos_inf", "neg_inf", "nan"])
def test_bw_unary_div_no_nan_zero_scalar_non_finite_grad(dtype, scalar, grad_value, device):
    shape = torch.Size([1, 1, 32, 32])
    grad = ttnn.from_torch(torch.full(shape, grad_value), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    inp = ttnn.from_torch(torch.ones(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.div_no_nan_bw(grad, inp, scalar)[0]).float()
    assert torch.equal(output, torch.zeros(shape)), f"expected 0.0 for scalar={scalar}, got {output[0, 0, 0, 0].item()}"
