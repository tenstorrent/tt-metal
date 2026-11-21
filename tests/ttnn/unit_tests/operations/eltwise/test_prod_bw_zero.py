# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Zero-input regression for ttnn.prod_bw (issue #54551).
#
# This file lives under tests/ttnn/unit_tests/ rather than tests/ttnn/nightly/ so that the
# PR pipeline executes it: tests/pipeline_reorg/ttnn_sanity_tests.yaml collects this
# directory ("ttnn eltwise group 1..4") on hardware and on the simulator, while the nightly
# eltwise path is only collected by tt-metal-l2-nightly.yaml.

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_pt_tt_prod,
)


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
@pytest.mark.parametrize(
    "input_layout",
    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ids=["tile", "row_major"],
)
def test_bw_prod_with_zeros(input_shapes, dim, num_zeros, input_layout, device):
    all_dimensions = dim is None

    torch.manual_seed(0)
    # +/-1 keeps the full product exact even when dim=None multiplies up to 12288 factors.
    # Non-unit magnitudes are covered by the single-dim test below.
    sign = torch.randint(0, 2, input_shapes).float() * 2 - 1
    in_data = sign.bfloat16()
    if all_dimensions:
        in_data.view(-1)[:num_zeros] = 0.0
    else:
        if input_shapes[dim] < num_zeros:
            pytest.skip(f"dim {dim} of {list(input_shapes)} is too short for {num_zeros} zeros")
        in_data.index_fill_(dim, torch.arange(num_zeros), 0.0)
    in_data = in_data.detach().clone().requires_grad_(True)

    input_tensor = ttnn.Tensor(in_data.detach().bfloat16(), ttnn.bfloat16).to(input_layout).to(device)
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

    # rtol=2e-2 allows up to 2% relative error, which covers the SFPU reciprocal. What this
    # assertion really pins is atol=0: the zero gradients, which the old formula got wrong,
    # have to come back exact.
    torch.testing.assert_close(result, in_data.grad.float(), rtol=2e-2, atol=0.0)


# The +/-1 matrix above makes every non-zero gradient +/-|grad|, so it cannot tell
# product(other factors) from its sign. Reducing along one dim keeps the product inside the
# bfloat16 range, which leaves room for real magnitudes.
POW2_MAGNITUDES = [2.0, 0.5, 4.0, 8.0, 0.25]


@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
@pytest.mark.parametrize("num_zeros", [0, 1, 2], ids=["no_zero", "one_zero", "two_zeros"])
def test_bw_prod_zero_keeps_other_factors(dim, num_zeros, device):
    input_shapes = torch.Size([4, 3, 32, 32])
    if input_shapes[dim] < num_zeros:
        pytest.skip(f"dim {dim} of {list(input_shapes)} is too short for {num_zeros} zeros")

    torch.manual_seed(0)
    sign = torch.randint(0, 2, input_shapes).float() * 2 - 1
    # Powers of two: the product along dim and every product/x quotient stay exact in bfloat16.
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
        # Guard the point of this test: some gradient must carry a magnitude, not just a sign.
        ratio = (in_data.grad.abs() / grad_data.abs().clamp(min=1e-30)).max()
        assert ratio > 1.5, f"expected gradients lost their magnitude (max ratio {ratio})"

    result = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()
    non_finite = (~torch.isfinite(result)).sum().item()
    assert non_finite == 0, (
        f"prod_bw returned {non_finite} non-finite gradient values for an input "
        f"with {num_zeros} zero(s) per reduced slice (dim={dim})"
    )

    # rtol=2e-2 allows up to 2% relative error, which covers the SFPU reciprocal. What this
    # assertion really pins is atol=0: the zero gradients, which the old formula got wrong,
    # have to come back exact.
    torch.testing.assert_close(result, in_data.grad.float(), rtol=2e-2, atol=0.0)
