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


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scalar", [1.0e-38, -8.0e-39, 0.0, 0.5])
def test_bw_div_no_nan_subnormal_scalar(scalar, dtype, device):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    mag = torch.rand(1, 1, 32, 32) * 1.5 + 0.5
    sign = torch.where(torch.rand(1, 1, 32, 32) < 0.5, -1.0, 1.0)
    grad_data = (sign * mag * 1.0e-37).to(torch_dtype)
    in_data = torch.rand(1, 1, 32, 32).to(torch_dtype)
    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.div_no_nan_bw(grad_tensor, input_tensor, scalar)[0]).float()
    s = torch.tensor(scalar, dtype=torch.float32)
    golden = torch.zeros_like(grad_data.float()) if scalar == 0.0 else grad_data.float() / s

    assert torch.allclose(output, golden, rtol=2e-2, atol=0.0)
