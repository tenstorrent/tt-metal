# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""rsqrt_bw at zero carries the sign the derivative has.

d/dx rsqrt(x) is -0.5 * x^-3/2, so at zero the gradient is -inf for a positive
incoming gradient and +inf for a negative one. The composite wrote +inf for both.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _t(v, dtype, device):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(
        torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("grad, want", [(2.0, float("-inf")), (-2.0, float("inf")), (0.5, float("-inf"))])
def test_rsqrt_bw_at_zero_follows_the_gradient_sign(device, dtype, grad, want):
    got = ttnn.to_torch(ttnn.rsqrt_bw(_t(grad, dtype, device), _t(0.0, dtype, device))[0])
    assert got.float().flatten()[0].item() == want


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("x, grad", [(4.0, 2.0), (0.25, -1.0), (1.0, 3.0)])
def test_rsqrt_bw_in_the_domain_matches_torch(device, dtype, x, grad):
    got = ttnn.to_torch(ttnn.rsqrt_bw(_t(grad, dtype, device), _t(x, dtype, device))[0]).float().flatten()[0].item()
    t = torch.tensor([x], requires_grad=True)
    torch.rsqrt(t).backward(torch.tensor([grad]))
    assert got == pytest.approx(t.grad.item(), rel=2e-2)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rsqrt_bw_out_of_domain_is_unchanged(device, dtype):
    got = ttnn.to_torch(ttnn.rsqrt_bw(_t(2.0, dtype, device), _t(-4.0, dtype, device))[0]).float().flatten()[0].item()
    # bfloat16 Dest cannot carry NaN and returns an infinity instead.
    assert got != got or got in (float("inf"), float("-inf"))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("x, grad", [(4.0, 2.0), (0.0, 2.0), (0.0, -2.0)])
def test_sqrt_bw_is_unchanged(device, dtype, x, grad):
    got = ttnn.to_torch(ttnn.sqrt_bw(_t(grad, dtype, device), _t(x, dtype, device))[0]).float().flatten()[0].item()
    t = torch.tensor([x], requires_grad=True)
    torch.sqrt(t).backward(torch.tensor([grad]))
    want = t.grad.item()
    assert got == pytest.approx(want, rel=2e-2) if abs(want) != float("inf") else got == want
