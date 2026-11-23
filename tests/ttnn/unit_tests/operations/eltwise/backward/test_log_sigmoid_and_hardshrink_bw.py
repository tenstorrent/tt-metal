# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _torch_dtype(dtype):
    return torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32


def _t(v, dtype, device):
    return ttnn.from_torch(torch.full(SHAPE, v, dtype=_torch_dtype(dtype)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _from_torch(value, dtype, device):
    return ttnn.from_torch(value.to(_torch_dtype(dtype)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _pcc(got, want):
    got = got.flatten().double()
    want = want.flatten().double()
    got = got - got.mean()
    want = want - want.mean()
    return (got @ want / (got.norm() * want.norm())).item()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "x, grad",
    [
        (-4.0, 2.0),
        (0.0, 2.0),
        (4.0, -1.5),
        # Large magnitude: the stable form has to saturate, not overflow. At
        # x = +80 the gradient is ~2e-35 and rounds to 0, at x = -80 it is grad.
        (-20.0, 2.0),
        (20.0, 2.0),
        (-80.0, 2.0),
        (80.0, 2.0),
    ],
)
def test_log_sigmoid_bw_matches_torch(device, dtype, x, grad):
    got = ttnn.to_torch(ttnn.log_sigmoid_bw(_t(grad, dtype, device), _t(x, dtype, device))[0]).float().flatten()[0].item()
    t = torch.tensor([x], dtype=torch.float32, requires_grad=True)
    torch.nn.functional.logsigmoid(t).backward(torch.tensor([grad], dtype=torch.float32))
    # abs tolerance, not just rel: on the saturating tail the expected value is
    # ~1e-35 here and exactly 0 on the device, which rel alone rejects.
    assert got == pytest.approx(t.grad.item(), rel=2e-2, abs=1e-4)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_bw_matches_torch_over_a_random_tensor(device, dtype):
    """Covers the whole range at once instead of four single elements: a
    saturation or overflow bug shows up as a correlation drop even when a
    handful of hand-picked points happen to pass."""
    torch.manual_seed(0)
    x = torch.empty(SHAPE, dtype=torch.float32).uniform_(-8.0, 8.0)
    grad = torch.empty(SHAPE, dtype=torch.float32).uniform_(-2.0, 2.0)
    t = x.clone().requires_grad_(True)
    torch.nn.functional.logsigmoid(t).backward(grad)
    got = ttnn.to_torch(
        ttnn.log_sigmoid_bw(_from_torch(grad, dtype, device), _from_torch(x, dtype, device))[0]
    ).float()
    # bf16 rounds the inputs to 8 mantissa bits before the op runs, so the
    # correlation is the meaningful assertion there; fp32 has to be near-exact.
    assert _pcc(got, t.grad.float()) > (0.999 if dtype == ttnn.float32 else 0.99)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_log_sigmoid_bw_propagates_nan(device, dtype):
    got = ttnn.to_torch(ttnn.log_sigmoid_bw(_t(2.0, dtype, device), _t(float("nan"), dtype, device))[0]).float().flatten()[0].item()
    # A NaN input has to come back non-finite. bfloat16 Dest cannot carry NaN
    # and returns an infinity instead, so accept either - same convention as
    # test_rsqrt_bw_zero_sign.py::test_rsqrt_bw_out_of_domain_is_unchanged.
    assert got != got or got in (float("inf"), float("-inf"))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "x, lambd, grad, want",
    [
        (0.0, 0.5, 3.0, 0.0),
        (1.0, 0.5, 3.0, 3.0),
        (-1.0, 0.5, 3.0, 3.0),
        (0.0, -0.5, 3.0, 3.0),
        (float("nan"), 0.5, 3.0, 0.0),
        # Large magnitude: the comparison must hold far outside the shrink band.
        (-20.0, 0.5, 3.0, 3.0),
        (20.0, 0.5, 3.0, 0.0),
        (-80.0, 0.5, 3.0, 3.0),
        (80.0, 0.5, 3.0, 0.0),
    ],
)
def test_hardshrink_bw_domain_and_nan(device, dtype, x, lambd, grad, want):
    got = ttnn.to_torch(ttnn.hardshrink_bw(_t(grad, dtype, device), _t(x, dtype, device), lambd=lambd)[0]).float().flatten()[0].item()
    assert got == pytest.approx(want, abs=1e-5)
