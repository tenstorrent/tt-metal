# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

OPS = [
    (ttnn.erfinv_bw, torch.erfinv, "inside"),
    (ttnn.erf_bw, torch.erf, "wide"),
    (ttnn.reciprocal_bw, torch.reciprocal, "wide"),
]


def inputs(where, dtype):
    torch.manual_seed(0)
    if where == "inside":
        return (torch.rand([1, 1, 32, 32]) * 1.8 - 0.9).to(dtype)
    return (torch.rand([1, 1, 32, 32]) * 8 - 4).to(dtype)


@pytest.mark.parametrize("ttnn_op, torch_op, domain", OPS)
@pytest.mark.parametrize("which_grad", ["ones", "mixed", "negative"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_backward_matches_torch(device, ttnn_op, torch_op, domain, which_grad, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = inputs(domain, torch_dtype)
    if which_grad == "ones":
        grad = torch.ones([1, 1, 32, 32], dtype=torch_dtype)
    elif which_grad == "negative":
        grad = -torch.ones([1, 1, 32, 32], dtype=torch_dtype)
    else:
        torch.manual_seed(1)
        grad = (torch.rand([1, 1, 32, 32]) * 2 - 1).to(torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(ttnn_op(dev(grad), dev(x))[0]).float()

    tx = x.float().clone().requires_grad_(True)
    torch_op(tx).backward(grad.float())

    keep = torch.isfinite(tx.grad) & torch.isfinite(got)
    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-3
    assert ((got[keep] - tx.grad[keep]).abs() / tx.grad[keep].abs().clamp(min=1.0)).max() < tol


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_erfinv_bw_at_the_boundary(device, dtype):
    # |x| == 1 takes sign(grad) * inf, |x| > 1 takes nan. Both boundaries were
    # written as a pair of tests against 1 and -1, which is abs(x) == 1.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    one = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    assert (ttnn.to_torch(ttnn.erfinv_bw(dev(one), dev(one))[0]).float() == float("inf")).all()
    assert (ttnn.to_torch(ttnn.erfinv_bw(dev(one), dev(-one))[0]).float() == float("inf")).all()
    assert (ttnn.to_torch(ttnn.erfinv_bw(dev(-one), dev(one))[0]).float() == float("-inf")).all()
    outside = ttnn.to_torch(ttnn.erfinv_bw(dev(one), dev(2 * one))[0]).float()
    # bfloat16 loses the nan through the pack and carries inf instead, before
    # this change as well.
    assert not torch.isfinite(outside).any()
    if dtype == ttnn.float32:
        assert outside.isnan().all()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_reciprocal_bw_at_zero(device, dtype):
    # x == 0 takes -sign(grad) * inf, and nan when the gradient is zero too.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    zero = torch.zeros([1, 1, 32, 32], dtype=torch_dtype)
    one = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    assert (ttnn.to_torch(ttnn.reciprocal_bw(dev(one), dev(zero))[0]).float() == float("-inf")).all()
    assert (ttnn.to_torch(ttnn.reciprocal_bw(dev(-one), dev(zero))[0]).float() == float("inf")).all()
    both_zero = ttnn.to_torch(ttnn.reciprocal_bw(dev(zero), dev(zero))[0]).float()
    assert not torch.isfinite(both_zero).any()
    if dtype == ttnn.float32:
        assert both_zero.isnan().all()
