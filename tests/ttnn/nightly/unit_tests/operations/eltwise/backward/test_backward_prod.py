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


# PCC can hide non-finite values, so check finiteness separately.
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([4, 3, 32, 32])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3, None],
)
@pytest.mark.parametrize("num_zeros", [0, 1, 2, 3], ids=["no_zero", "one_zero", "two_zeros", "three_zeros"])
def test_bw_prod_with_zeros(input_shapes, dim, num_zeros, device):
    all_dimensions = dim is None

    torch.manual_seed(0)
    # +/-1 keeps the full product exact. The zeros inserted below are the only zeros.
    sign = torch.randint(0, 2, input_shapes).float() * 2 - 1
    in_data = sign.bfloat16()
    if all_dimensions:
        in_data.view(-1)[:num_zeros] = 0.0
    else:
        if input_shapes[dim] < num_zeros:
            pytest.skip(f"dim {dim} of {list(input_shapes)} is too short for {num_zeros} zeros")
        in_data.index_fill_(dim, torch.arange(num_zeros), 0.0)
    in_data = in_data.detach().clone().requires_grad_(True)

    input_tensor = ttnn.Tensor(in_data.detach().bfloat16(), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device, all_dimensions, dim)

    if all_dimensions:
        pyt_y = torch.prod(in_data)
        tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor)
    else:
        pyt_y = torch.prod(in_data, dim=dim, keepdim=True)
        tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor, dim=dim)

    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    result = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()
    non_finite = (~torch.isfinite(result)).sum().item()
    assert non_finite == 0, (
        f"prod_bw returned {non_finite} non-finite gradient values for an input "
        f"with {num_zeros} zero(s) per reduced slice (dim={dim})"
    )

    # SFPU reciprocal(+/-1) may differ by one bfloat16 step; zero gradients must stay exact.
    torch.testing.assert_close(result, in_data.grad.float(), rtol=2e-2, atol=0.0)
