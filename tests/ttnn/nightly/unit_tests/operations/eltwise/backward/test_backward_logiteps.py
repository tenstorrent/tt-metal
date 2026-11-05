# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
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
@pytest.mark.parametrize(
    "eps",
    (2, -0.001, 0.4, 0.5, 1.0),
)
def test_bw_logiteps(input_shapes, eps, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -2, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    tt_output_tensor_on_device = ttnn.logiteps_bw(grad_tensor, input_tensor, eps=eps)
    golden_function = ttnn.get_golden_function(ttnn.logiteps_bw)
    golden_tensor = golden_function(grad_data, in_data, eps)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_logiteps_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -2, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    tt_output_tensor_on_device = ttnn.logiteps_bw(grad_tensor, input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.logiteps_bw)
    golden_tensor = golden_function(grad_data, in_data)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("eps", [-1.0, -0.25, 0.125])
def test_bw_logiteps_domain(eps, dtype, device):
    vals = torch.tensor([-0.75, -0.5, -0.125, 0.0625, 0.25, 0.5, 0.75, 0.9375, 1.125, 1.5, 1.75])
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    in_data = vals.repeat(32 * 32 // len(vals) + 1)[: 32 * 32].reshape(1, 1, 32, 32).to(torch_dtype)
    grad_data = torch.ones_like(in_data)
    input_tensor = ttnn.from_torch(in_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.logiteps_bw(grad_tensor, input_tensor, eps=eps)[0]).float()
    x = in_data.float().requires_grad_(True)
    torch.logit(x, eps=eps).backward(grad_data.float())
    golden = x.grad
    if dtype == ttnn.bfloat16:
        golden = torch.where(torch.isnan(golden), torch.inf, golden)

    assert torch.equal(torch.isnan(output), torch.isnan(golden))
    finite = torch.isfinite(golden)
    assert torch.equal(torch.isfinite(output), finite)
    assert torch.allclose(output[finite], golden[finite], rtol=2e-2, atol=1e-3)
