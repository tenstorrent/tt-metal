# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

OPS = [
    (ttnn.logaddexp_bw, torch.logaddexp),
    (ttnn.logaddexp2_bw, torch.logaddexp2),
    (ttnn.ldexp_bw, torch.ldexp),
]


def spread(name, dtype):
    torch.manual_seed(0)
    if name == "ordinary":
        return (torch.rand([1, 1, 32, 32]) * 10 - 5), (torch.rand([1, 1, 32, 32]) * 10 - 5)
    if name == "far apart":
        return (torch.rand([1, 1, 32, 32]) * 10 + 40), (torch.rand([1, 1, 32, 32]) * 10 - 50)
    a = torch.rand([1, 1, 32, 32]) * 4 - 2
    return a, a.clone()


@pytest.mark.parametrize("ttnn_op, torch_op", OPS)
@pytest.mark.parametrize("case", ["ordinary", "far apart", "equal"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_backward_matches_torch(device, ttnn_op, torch_op, case, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a32, b32 = spread(case, dtype)
    a, b = a32.to(torch_dtype), b32.to(torch_dtype)
    grad = (torch.rand([1, 1, 32, 32]) * 2 - 1).to(torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got_a, got_b = [ttnn.to_torch(t).float() for t in ttnn_op(dev(grad), dev(a), dev(b))]

    ta = a.float().clone().requires_grad_(True)
    tb = b.float().clone().requires_grad_(True)
    torch_op(ta, tb).backward(grad.float())

    tol = 0.05 if dtype == ttnn.bfloat16 else 1e-4
    keep = torch.isfinite(ta.grad) & torch.isfinite(got_a)
    assert ((got_a[keep] - ta.grad[keep]).abs() / ta.grad[keep].abs().clamp(min=1.0)).max() < tol
    keep = torch.isfinite(tb.grad) & torch.isfinite(got_b)
    assert ((got_b[keep] - tb.grad[keep]).abs() / tb.grad[keep].abs().clamp(min=1.0)).max() < tol
