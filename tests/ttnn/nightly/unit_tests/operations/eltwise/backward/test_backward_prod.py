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


# PCC can hide non-finite values, so finiteness is checked separately before the
# comparison against torch autograd (issue #54551).
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
def test_bw_prod_zero_inputs(input_shapes, dim, num_zeros, device):
    """prod_bw must return finite gradients matching autograd when the reduced input holds zeros.

    dy/dx_i = grad * prod_{j != i} x_j: no zeros -> prod*grad/x; exactly one zero ->
    grad*prod(non-zeros) at that position and 0 elsewhere; two or more zeros -> 0 everywhere.
    """
    all_dimensions = dim is None
    torch.manual_seed(0)
    # +/-1 keeps the total product exact in bfloat16 even for dim=None over 12288 factors.
    in_data = (torch.randint(0, 2, input_shapes).float() * 2 - 1).bfloat16()
    if all_dimensions:
        in_data.view(-1)[:num_zeros] = 0.0
    else:
        if input_shapes[dim] < num_zeros:
            pytest.skip(f"dim {dim} of {list(input_shapes)} too short for {num_zeros} zeros")
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
        f"prod_bw returned {non_finite} non-finite gradient values with {num_zeros} zero(s) "
        f"per reduced slice (dim={dim})"
    )
    # Zero gradients must stay exact; SFPU reciprocal may differ by ~1 bfloat16 step elsewhere.
    torch.testing.assert_close(result, in_data.grad.float(), rtol=2e-2, atol=0.0)


# +/-1 data above cannot verify that the single-zero position carries grad * prod(other
# factors) rather than just any finite value; use powers of two so every product of others
# is exact in bfloat16 along a single reduced dimension.
POW2_MAGNITUDES = [2.0, 0.5, 4.0, 8.0, 0.25]


@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
@pytest.mark.parametrize("num_zeros", [0, 1, 2], ids=["no_zero", "one_zero", "two_zeros"])
def test_bw_prod_zero_keeps_other_factors(dim, num_zeros, device):
    input_shapes = torch.Size([4, 3, 32, 32])
    if input_shapes[dim] < num_zeros:
        pytest.skip(f"dim {dim} of {list(input_shapes)} too short for {num_zeros} zeros")

    torch.manual_seed(0)
    sign = (torch.randint(0, 2, input_shapes).float() * 2 - 1).bfloat16()
    index = torch.arange(input_shapes[dim]).view([-1 if k == dim % 4 else 1 for k in range(4)])
    magnitude = torch.tensor(POW2_MAGNITUDES)[index % len(POW2_MAGNITUDES)].expand(input_shapes)
    in_data = (sign * magnitude).bfloat16().clone()
    in_data.index_fill_(dim, torch.arange(num_zeros), 0.0)
    in_data = in_data.detach().clone().requires_grad_(True)

    input_tensor = ttnn.Tensor(in_data.detach().bfloat16(), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    grad_data, grad_tensor = data_gen_pt_tt_prod(input_shapes, device, False, dim)

    pyt_y = torch.prod(in_data, dim=dim, keepdim=True)
    tt_output_tensor_on_device = ttnn.prod_bw(grad_tensor, input_tensor, dim=dim)
    in_data.retain_grad()
    pyt_y.backward(gradient=grad_data)

    if num_zeros < 2:
        # Guard the point of this test: the gradient at the zero must carry a magnitude.
        ratio = (in_data.grad.abs() / grad_data.abs().clamp(min=1e-30)).max()
        assert ratio > 1.5, f"expected gradients lost their magnitude (max ratio {ratio})"

    result = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()
    non_finite = (~torch.isfinite(result)).sum().item()
    assert non_finite == 0, (
        f"prod_bw returned {non_finite} non-finite gradient values with {num_zeros} zero(s) "
        f"per reduced slice (dim={dim})"
    )
    torch.testing.assert_close(result, in_data.grad.float(), rtol=2e-2, atol=0.0)
