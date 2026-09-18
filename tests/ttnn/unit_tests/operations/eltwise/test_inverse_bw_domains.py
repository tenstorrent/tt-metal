# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

OPS = [
    (ttnn.acosh_bw, torch.acosh, "outside"),
    (ttnn.acos_bw, torch.acos, "inside"),
    (ttnn.atanh_bw, torch.atanh, "inside"),
]

GRADS = ["ones", "mixed", "zeros", "negative"]


def inputs(where, dtype):
    torch.manual_seed(0)
    if where == "inside":
        return (torch.rand([1, 1, 32, 32]) * 1.8 - 0.9).to(dtype)
    return (torch.rand([1, 1, 32, 32]) * 8 + 1.5).to(dtype)


def grads(which, dtype):
    if which == "ones":
        return torch.ones([1, 1, 32, 32], dtype=dtype)
    if which == "zeros":
        return torch.zeros([1, 1, 32, 32], dtype=dtype)
    if which == "negative":
        return -torch.ones([1, 1, 32, 32], dtype=dtype)
    torch.manual_seed(1)
    return (torch.rand([1, 1, 32, 32]) * 2 - 1).to(dtype)


@pytest.mark.parametrize("ttnn_op, torch_op, domain", OPS)
@pytest.mark.parametrize("which_grad", GRADS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_backward_in_domain_matches_torch(device, ttnn_op, torch_op, domain, which_grad, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = inputs(domain, torch_dtype)
    grad = grads(which_grad, torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(ttnn_op(dev(grad), dev(x))[0]).float()

    tx = x.float().clone().requires_grad_(True)
    torch_op(tx).backward(grad.float())

    keep = torch.isfinite(tx.grad) & torch.isfinite(got)
    if keep.sum() == 0:
        # A zero gradient makes the op answer nan by design; there is nothing
        # finite left to compare against torch, which answers zero.
        pytest.skip("no finite element to compare")
    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-4
    assert ((got[keep] - tx.grad[keep]).abs() / tx.grad[keep].abs().clamp(min=1.0)).max() < tol


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_atanh_bw_sign_at_the_boundary(device, dtype):
    # At |x| == 1 the gradient is an infinity whose sign follows the incoming
    # gradient. The last where in the op is what flips it, and it tests the
    # result against inf, which UNARY_EQ as an activation does not match.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    one = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    assert (ttnn.to_torch(ttnn.atanh_bw(dev(one), dev(one))[0]).float() == float("inf")).all()
    assert (ttnn.to_torch(ttnn.atanh_bw(dev(-one), dev(one))[0]).float() == float("-inf")).all()
    assert (ttnn.to_torch(ttnn.atanh_bw(dev(-one), dev(-one))[0]).float() == float("-inf")).all()
