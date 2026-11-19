# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_pt_tt,
    data_gen_pt_tt_prod,
    compare_results,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),  # 0
        (torch.Size([1, 1, 320, 384])),  # 1
        (torch.Size([4, 2, 32, 32])),  # 2
        (torch.Size([1, 3, 320, 384])),  # 3
        (torch.Size([4, 3, 32, 32])),  # 4
        (torch.Size([4, 3, 64, 64])),  # 5
        (torch.Size([4, 3, 320, 320])),  # 6
        (torch.Size([4, 3, 32, 32])),  # 7
        (torch.Size([1, 3, 320, 320])),  # 8
        (torch.Size([1, 4, 320, 384])),  # 9
        (torch.Size([4, 4, 32, 32])),  # 10
        (torch.Size([5, 4, 32, 32])),  # 11
        (torch.Size([6, 4, 32, 32])),  # 12
        (torch.Size([4, 5, 32, 32])),  # 13
        (torch.Size([4, 6, 32, 32])),  # 14
        (torch.Size([4, 10, 32, 32])),  # 15
        (torch.Size([4, 20, 32, 32])),  # 16
        (torch.Size([4, 30, 32, 32])),  # 17
        (torch.Size([4, 31, 32, 32])),  # 18
        (torch.Size([4, 32, 32, 32])),  # 19
        (torch.Size([4, 33, 32, 32])),  # 20
        (torch.Size([4, 63, 32, 32])),  # 21
        (torch.Size([4, 64, 32, 32])),  # 22
        (torch.Size([32, 64, 64, 64])),  # 23
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3, None],
)
def test_bw_prod(input_shapes, dim, device):
    all_dimensions = dim is None
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device, all_dimensions, dim)

    if all_dimensions:
        pyt_y = torch.prod(in_data)
        tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor)
    else:
        pyt_y = torch.prod(in_data, dim=dim, keepdim=True)
        tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor, dim=dim)

    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([32, 64, 64, 64])),
    ),
)
def test_bw_prod_default_both(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device)
    pyt_y = torch.prod(in_data)
    tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize("full_grad", [False, True])
@pytest.mark.parametrize("dim", [-1, -2])
def test_prod_bw_logical_shape_not_tile_aligned(device, dim, full_grad):
    torch.manual_seed(0)
    x = (1.0 + 0.05 * torch.randn((1, 1, 40, 50))).bfloat16().float().requires_grad_(True)
    y = torch.prod(x, dim=dim, keepdim=True)
    grad = torch.randn(y.shape).bfloat16().float()
    y.backward(grad)
    tt_grad_in = grad
    if full_grad:
        tt_grad_in = torch.randn(x.shape).bfloat16().float()
        if dim == -1:
            tt_grad_in[..., :1] = grad
        else:
            tt_grad_in[..., :1, :] = grad
    tt_x = ttnn.from_torch(x.detach().bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_grad = ttnn.from_torch(tt_grad_in.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.prod_bw(tt_grad, tt_x, dim=dim)[0]).float()
    assert list(out.shape) == list(x.shape)
    assert_with_pcc(x.grad, out, 0.999)
