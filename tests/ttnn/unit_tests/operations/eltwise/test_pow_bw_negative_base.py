# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Issue #54439: pow_bw returned +inf for every negative base.

d/dx x^n = n * x^(n-1) is finite for every finite base and every integral n, but the op formed
the power as exp((n-1) * log(x)), whose log is NaN for x < 0, and a where(lez(input), +inf, ...)
painted over that NaN. The reference here is torch.autograd rather than ttnn's golden function,
so a regression in either one fails this.
"""

import pytest
import torch

import ttnn


def _torch_grad(base, exponent, dtype):
    x = torch.tensor(base, dtype=dtype).reshape(1, 1, 1, 1).expand(1, 1, 32, 32).clone()
    x.requires_grad_(True)
    torch.pow(x, exponent).backward(gradient=torch.ones_like(x))
    return x.grad


def _device_grad(base, exponent, ttnn_dtype, torch_dtype, device):
    x = torch.tensor(base, dtype=torch_dtype).reshape(1, 1, 1, 1).expand(1, 1, 32, 32).contiguous()
    grad = torch.ones_like(x)
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.pow_bw(to_dev(grad), to_dev(x), exponent)
    return ttnn.to_torch(out[0])


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)])
@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("base", [-2.0, -0.5, 0.0, 0.5, 2.0], ids=lambda b: f"base{b}")
def test_pow_bw_integral_exponent_matches_autograd(base, exponent, torch_dtype, ttnn_dtype, device):
    """An integral exponent has a finite gradient at every finite base, negative ones included."""
    expected = _torch_grad(base, exponent, torch_dtype)
    actual = _device_grad(base, exponent, ttnn_dtype, torch_dtype, device)

    assert torch.isfinite(actual).all(), f"pow_bw({base}, {exponent}) returned {actual.flatten()[0].item()}"
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.float32, ttnn.float32)])
@pytest.mark.parametrize("exponent", [2.5])
def test_pow_bw_non_integral_exponent_is_nan_below_zero(exponent, torch_dtype, ttnn_dtype, device):
    """x^2.5 is not defined for x < 0, so the gradient there is NaN as in torch, never +inf.

    float32 only: a bfloat16 destination cannot carry a NaN, which collapses it to an infinity
    on the way out and makes the distinction unobservable at that dtype.
    """
    actual = _device_grad(-2.0, exponent, ttnn_dtype, torch_dtype, device)
    assert torch.isnan(actual).all(), f"expected NaN, got {actual.flatten()[0].item()}"

    positive = _device_grad(2.0, exponent, ttnn_dtype, torch_dtype, device)
    torch.testing.assert_close(
        positive, _torch_grad(2.0, exponent, torch_dtype).to(positive.dtype), rtol=1e-2, atol=1e-2
    )
