# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""rsqrt_bw at zero carries the sign the derivative has.

d/dx rsqrt(x) is -0.5 * x^-3/2, so at zero the gradient is -inf for a positive
incoming gradient and +inf for a negative one. The composite wrote +inf for both.
Negative subnormal inputs flush to zero in the SFPU while torch reads them as
negative numbers (NaN for the rsqrt and sqrt gradients, a large finite value for
the log ones), so they keep the +0 pole; only the exact -0.0 bit pattern (0x8000)
flips it.
"""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import run_for_wormhole_b0_or_blackhole

SHAPE = (1, 1, 32, 32)
NUMEL = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3]
# The fused BF16 kernels of the reciprocal derivative take the pole sign from the input zero.
POLE_OPS = ["rsqrt_bw", "sqrt_bw", "log_bw", "log10_bw", "log2_bw"]


def _t(v, dtype, device):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


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


def _pole_pair(x_values, grad_values, device):
    x = torch.tensor(x_values, dtype=torch.bfloat16).reshape(SHAPE)
    grad = torch.tensor(grad_values, dtype=torch.bfloat16).reshape(SHAPE)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x, grad, tx, tg


def _pole_reference(op, x, grad):
    x64 = x.to(torch.float64).clone().requires_grad_(True)
    getattr(torch, op.removesuffix("_bw"))(x64).backward(grad.to(torch.float64))
    return x64.grad.detach().to(torch.bfloat16)


def _assert_pole_sign(op, x, grad, tx, tg):
    got = ttnn.to_torch(getattr(ttnn, op)(tg, tx)[0]).to(torch.bfloat16)
    want = _pole_reference(op, x, grad)
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


# Both signs of the input zero, with both signs of a finite or infinite gradient. rsqrt takes the
# opposite sign from the log family at the same zero, so one shared reference checks each op.
@run_for_wormhole_b0_or_blackhole("the fused BF16 kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("op", POLE_OPS)
@pytest.mark.parametrize(
    "grad",
    [1.0, -1.0, float("inf"), float("-inf")],
    ids=["pos", "neg", "pos_inf", "neg_inf"],
)
def test_bw_negative_zero_pole_follows_the_input_sign(device, op, grad):
    x, g, tx, tg = _pole_pair([-0.0] * NUMEL, [grad] * NUMEL, device)
    _assert_pole_sign(op, x, g, tx, tg)


@run_for_wormhole_b0_or_blackhole("the fused BF16 kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("op", POLE_OPS)
def test_bw_mixed_zero_pole_follows_each_lane(device, op):
    pattern_x = [-0.0, 0.0, -0.0, 0.0]
    pattern_g = [1.0, 1.0, -1.0, -1.0]
    repeats = NUMEL // len(pattern_x) + 1
    x, g, tx, tg = _pole_pair((pattern_x * repeats)[:NUMEL], (pattern_g * repeats)[:NUMEL], device)
    _assert_pole_sign(op, x, g, tx, tg)


def _bits_pair(bits_values, grad_values, device):
    """A pole pair whose inputs are raw BF16 bit patterns, stored through int16 like the nightly
    suites store theirs."""
    signed = [b - 0x10000 if b >= 0x8000 else b for b in bits_values]
    x = torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16).reshape(SHAPE)
    grad = torch.tensor(grad_values, dtype=torch.bfloat16).reshape(SHAPE)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x, grad, tx, tg


def _flushed_pole_reference(op, grad):
    """The nightly exhaustive suites accept a lane recomputed with its subnormal input flushed to +0
    (the smallest normal and below becomes zero), beside torch's own answer for the operand."""
    return _pole_reference(op, torch.zeros(SHAPE, dtype=torch.bfloat16), grad)


# torch reads a negative subnormal (bits 0x8001 to 0x807F) as a negative number: the rsqrt and sqrt
# gradients are NaN there, while the log gradients stay finite because 1/x is defined across the
# negative axis. The kernel flushes the pattern to zero like +0, and the nightly exhaustive suites
# accept either of the two torch recomputes (on the operand, or with the subnormal flushed to +0), so
# it returns the +0 pole; the signed -0 pole of the opposite sign is the only wrong answer. Only
# nonzero finite gradients are used here: a zero gradient keeps its separate pre-existing infinity.
@run_for_wormhole_b0_or_blackhole("the fused BF16 kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("op", POLE_OPS)
@pytest.mark.parametrize("bits", [0x8001, 0x807F], ids=["smallest", "largest"])
@pytest.mark.parametrize("grad", [1.0, -1.0, 2.5, -2.5], ids=["pos", "neg", "pos_big", "neg_big"])
def test_bw_negative_subnormal_keeps_the_positive_zero_pole(device, op, bits, grad):
    x, g, tx, tg = _bits_pair([bits] * NUMEL, [grad] * NUMEL, device)
    got = ttnn.to_torch(getattr(ttnn, op)(tg, tx)[0]).to(torch.bfloat16)
    torch_answer = _pole_reference(op, x, g)
    flushed = _flushed_pole_reference(op, g)
    if op in ("rsqrt_bw", "sqrt_bw"):
        # Their gradients are NaN at every negative operand, so torch's own answer is NaN here.
        assert torch_answer.isnan().all(), f"torch gives {torch_answer.flatten()[0].item()} at bits 0x{bits:04X}"
    allowed = got.isnan() | (got == flushed) | (got == torch_answer)
    assert allowed.all(), (
        f"{op} at bits 0x{bits:04X}, grad {grad}: got {got.flatten()[0].item()}, "
        f"torch {torch_answer.flatten()[0].item()}, flushed pole {flushed.flatten()[0].item()}"
    )


@run_for_wormhole_b0_or_blackhole("the fused BF16 kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("op", POLE_OPS)
def test_bw_negative_subnormal_lanes_beside_zeros(device, op):
    """The exact -0 pattern shares a tile with negative subnormals: the -0 lanes take torch's signed
    pole, the subnormal lanes keep the +0 pole or one of torch's two recomputes, lane by lane."""
    pattern_bits = [0x8000, 0x8001, 0x8000, 0x8040]
    pattern_g = [1.0, 1.0, -1.0, -1.0]
    repeats = NUMEL // len(pattern_bits) + 1
    x, g, tx, tg = _bits_pair((pattern_bits * repeats)[:NUMEL], (pattern_g * repeats)[:NUMEL], device)
    got = ttnn.to_torch(getattr(ttnn, op)(tg, tx)[0]).to(torch.bfloat16)
    exact = _pole_reference(op, x, g)
    flushed = _flushed_pole_reference(op, g)
    subnormal = (x.view(torch.int16) & 0x7FFF) != 0
    allowed = torch.where(subnormal, got.isnan() | (got == flushed) | (got == exact), got == exact)
    assert allowed.all(), (
        f"{op}: got {got.flatten()[0].item()}, torch {exact.flatten()[0].item()}, "
        f"flushed pole {flushed.flatten()[0].item()}"
    )
