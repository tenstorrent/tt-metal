# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

NAN = float("nan")


def run(device, dtype, grad, a, b):
    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return [ttnn.to_torch(t).float() for t in ttnn.xlogy_bw(dev(grad), dev(a), dev(b))]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_bw_matches_torch(device, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    a = (torch.rand([1, 1, 32, 32]) * 10 - 5).to(torch_dtype)
    b = (torch.rand([1, 1, 32, 32]) * 10 + 0.5).to(torch_dtype)
    grad = (torch.rand([1, 1, 32, 32]) * 2 - 1).to(torch_dtype)

    got_a, got_b = run(device, dtype, grad, a, b)

    ta = a.float().clone().requires_grad_(True)
    tb = b.float().clone().requires_grad_(True)
    torch.xlogy(ta, tb).backward(grad.float())

    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-4
    assert (got_a - ta.grad).abs().max() < tol
    assert ((got_b - tb.grad).abs() / tb.grad.abs().clamp(min=1.0)).max() < tol


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_bw_a_zero_b_not_positive(device, dtype):
    # The logical_and of eqz(a) and le(b, 0) selects zero for grad_a here, and
    # the ltz(b) below it selects nan where only b is negative.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    zero = torch.zeros([1, 1, 32, 32], dtype=torch_dtype)
    one = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    got_a, _ = run(device, dtype, one, zero, -one)
    assert (got_a == 0).all()

    got_a, _ = run(device, dtype, one, one, -one)
    assert not torch.isfinite(got_a).any()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_xlogy_bw_b_zero(device, dtype):
    # grad_b is sign(grad) * inf where b is zero.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    zero = torch.zeros([1, 1, 32, 32], dtype=torch_dtype)
    a = torch.full([1, 1, 32, 32], 2.0, dtype=torch_dtype)
    grad = torch.full([1, 1, 32, 32], -1.0, dtype=torch_dtype)

    _, got_b = run(device, dtype, grad, a, zero)
    assert (got_b == float("-inf")).all()
