# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def reference(x, y):
    xd, yd = x.double(), y.double()
    s = xd * xd + yd * yd
    return (yd / s).float(), (-xd / s).float()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scale", [1.0, 1e-18, 1e18])
def test_atan2_bw_magnitudes(device, dtype, scale):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = ((torch.rand([1, 1, 32, 32]) * 20 - 10) * scale).to(torch_dtype)
    y = ((torch.rand([1, 1, 32, 32]) * 20 - 10) * scale).to(torch_dtype)
    grad = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got_a, got_b = [ttnn.to_torch(t).float() for t in ttnn.atan2_bw(dev(grad), dev(x), dev(y))]
    want_a, want_b = reference(x.float(), y.float())

    # Only where the reference itself survives the format: at 1e18 and 1e-18 the
    # sum of squares saturates, which it did before this change as well.
    keep = torch.isfinite(want_a) & torch.isfinite(got_a) & (want_a.abs() > 0) & (got_a.abs() > 0)
    if keep.sum() == 0:
        pytest.skip("no element of this set is representable")
    rel_a = ((got_a[keep] - want_a[keep]) / want_a[keep]).abs()
    rel_b = ((got_b[keep] - want_b[keep]) / want_b[keep]).abs()
    # Away from unit scale the sum of squares is close to the end of the format
    # and the surviving elements carry more error, which they did before too.
    tol = 0.05 if (dtype == ttnn.bfloat16 or scale != 1.0) else 1e-3
    assert rel_a.mean() < tol and rel_b.mean() < tol


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_atan2_bw_origin_is_not_finite(device, dtype):
    # Both gradients are the nan the where writes in where a and b are both
    # zero, which is what the logical_and of the two eqz selects. In bfloat16
    # that nan reaches the tensor as inf, since bfloat16 loses nan through the
    # pack; this is the case before this change as well.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    zero = torch.zeros([1, 1, 32, 32], dtype=torch_dtype)
    one = torch.ones([1, 1, 32, 32], dtype=torch_dtype)

    def dev(t):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got_a, got_b = [ttnn.to_torch(t).float() for t in ttnn.atan2_bw(dev(one), dev(zero), dev(zero))]
    assert not torch.isfinite(got_a).any() and not torch.isfinite(got_b).any()
    if dtype == ttnn.float32:
        assert got_a.isnan().all() and got_b.isnan().all()

    # One of the two being zero is an ordinary point, not the origin.
    got_a, got_b = [ttnn.to_torch(t).float() for t in ttnn.atan2_bw(dev(one), dev(zero), dev(one))]
    assert not got_a.isnan().any() and not got_b.isnan().any()
