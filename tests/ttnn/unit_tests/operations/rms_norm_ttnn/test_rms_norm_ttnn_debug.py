# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for rms_norm_ttnn.

DO NOT DELETE.  These are the hand-calculable inputs the implementation was
brought up against, and they are the first thing to reach for if a numerical
mismatch ever comes back: every intermediate value is a constant you can write
down, so a DEVICE_PRINT trace can be compared against arithmetic rather than
against another (possibly equally wrong) run.

The op is five stages, and each pattern below is chosen to separate them:

    1. t = x + residual        BEFORE the statistics
    2. m = mean(t^2, dim=-1)
    3. y = t / sqrt(m + eps)
    4. y = y * weight
    5. y = y + bias            AFTER the scale

Patterns and what each one catches:

  all-ones        every intermediate is 1: mean(t^2) = 1, rsqrt(1 + eps) ~ 1,
                  so the output is `weight + bias` exactly.  A dropped operand
                  shows up as an integer-sized error, not a rounding one.
  residual = -x   cancels the activation, so epsilon is the WHOLE denominator.
                  Catches (a) statistics taken over x instead of x + r, and
                  (b) epsilon dropped from the denominator (-> NaN).
  monotonic       every element is unique, so any tilize / untilize / chunk
                  reordering is visible as a permutation rather than as noise.
  position-coded  t[r][c] = r * 100 + c: each element encodes where it came
                  from, which is what makes a row/column swap readable.
  bias >> value   a bias two orders above the normalized value: catches a bias
                  applied BEFORE the normalize, or folded into the scale.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


def _dev(t, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _vec(width, value, device, *, layout):
    return _dev(torch.full((1, 1, 1, width), value, dtype=torch.bfloat16), device, layout=layout)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("with_weight, with_bias, with_residual", [(False, False, False), (True, True, False)])
def test_all_ones(device, layout, with_weight, with_bias, with_residual):
    """All-ones input: mean(t^2) = 1, so out = weight + bias, exactly.

    Hand-calculation, no residual:
        t = 1  ->  mean(t^2) = 1  ->  y = 1 / sqrt(1 + 1e-12) = 1.0
        * weight(2.0) = 2.0   + bias(-0.5) = 1.5
    """
    shape = (1, 1, 64, 128)
    width = shape[-1]
    x = torch.ones(shape, dtype=torch.bfloat16)
    kwargs = {}
    if with_weight:
        kwargs["weight"] = _vec(width, 2.0, device, layout=layout)
    if with_bias:
        kwargs["bias"] = _vec(width, -0.5, device, layout=layout)
    if with_residual:
        kwargs["residual_input_tensor"] = _dev(torch.ones(shape, dtype=torch.bfloat16), device, layout=layout)

    out = ttnn.to_torch(rms_norm_ttnn(_dev(x, device, layout=layout), epsilon=1e-12, **kwargs)).float()

    expected = 1.0
    if with_weight:
        expected *= 2.0
    if with_bias:
        expected += -0.5
    assert torch.allclose(out, torch.full_like(out, expected), rtol=0.02, atol=0.02), (
        f"all-ones expected {expected} everywhere; max deviation "
        f"{(out - expected).abs().max().item()} (min {out.min().item()}, max {out.max().item()})"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_ones_residual_doubles_nothing(device, layout):
    """t = 1 + 1 = 2, and mean(t^2) = 4, so y = 2 / 2 = 1 -- NOT 2.

    The residual therefore cannot be detected by the output's magnitude, which
    is exactly why the cancellation test below exists.  What this DOES catch is
    a residual added AFTER the normalize (which would give 1 + 1 = 2).
    """
    shape = (1, 1, 32, 64)
    ones = torch.ones(shape, dtype=torch.bfloat16)
    out = ttnn.to_torch(
        rms_norm_ttnn(
            _dev(ones, device, layout=layout),
            epsilon=1e-12,
            residual_input_tensor=_dev(ones, device, layout=layout),
        )
    ).float()
    assert torch.allclose(out, torch.ones_like(out), rtol=0.02, atol=0.02), (
        f"norm(1 + 1) must be 1 (t = 2, rms = 2); got min {out.min().item()} max {out.max().item()} -- "
        f"a value near 2 means the residual was added after the normalize"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("epsilon", [1e-5, 1e-12])
def test_residual_cancels_to_zero_not_nan(device, layout, epsilon):
    """residual = -x leaves an all-zero row: epsilon is the whole denominator.

    0 / sqrt(0 + eps) = 0 for every eps > 0.  Two distinct faults land here:
    statistics over x instead of x + r (the output would be finite but nonzero),
    and epsilon dropped from the denominator (0/0 -> NaN).
    """
    shape = (1, 1, 32, 64)
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32)
    out = ttnn.to_torch(
        rms_norm_ttnn(
            _dev(x, device, dtype=ttnn.float32, layout=layout),
            epsilon=epsilon,
            residual_input_tensor=_dev(-x, device, dtype=ttnn.float32, layout=layout),
        )
    )
    assert torch.isfinite(out).all(), "a cancelled residual must not produce NaN/Inf"
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=1e-6)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_monotonic_is_not_reordered(device, layout):
    """arange input: unique values, so a permutation is visible element-wise.

    Compared against the exported torch reference rather than a closed form --
    the point of the pattern is the ORDER, and an elementwise diff over unique
    values localizes a reorder to the exact position.
    """
    shape = (1, 1, 64, 96)
    x = torch.arange(shape[-2] * shape[-1], dtype=torch.float32).reshape(shape) / 1000.0
    ttnn_out = rms_norm_ttnn(_dev(x, device, dtype=ttnn.float32, layout=layout), epsilon=1e-12)
    out = ttnn.to_torch(ttnn_out).float()
    expected = torch_rms_norm_ttnn(x, epsilon=1e-12).float()
    diff = (out - expected).abs()
    worst = diff.flatten().argmax().item()
    assert diff.max() < 2e-2, (
        f"monotonic input diverged at flat index {worst} "
        f"(got {out.flatten()[worst].item()}, expected {expected.flatten()[worst].item()}); "
        f"a large diff at a regular stride means a tile/chunk reorder"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_position_coded_rows_stay_independent(device, layout):
    """t[r][c] = r * 100 + c: each row's statistic is different by construction.

    RMSNorm normalizes each row in ISOLATION, so a stat that leaked across rows
    (a cb_row_stat ring straddle, a mis-indexed Col broadcast) changes rows that
    were computed correctly on their own.  Checked per row against the reference.
    """
    rows, width = 96, 64
    x = torch.zeros(1, 1, rows, width, dtype=torch.float32)
    for r in range(rows):
        x[0, 0, r, :] = r * 100 + torch.arange(width, dtype=torch.float32)
    out = ttnn.to_torch(rms_norm_ttnn(_dev(x, device, dtype=ttnn.float32, layout=layout), epsilon=1e-12)).float()
    expected = torch_rms_norm_ttnn(x, epsilon=1e-12).float()
    per_row = (out - expected).abs().amax(dim=-1).flatten()
    bad = (per_row > 2e-2).nonzero().flatten().tolist()
    assert not bad, f"rows {bad[:8]} diverged (max per-row diff {per_row.max().item()}) -- a stat leaked across rows"


def test_bias_dominates_the_normalized_value(device):
    """A bias two orders above the normalized value.

    out = y * w + b with |b| ~ 100 and |y * w| ~ 1, so a bias applied BEFORE the
    normalize (which would divide it out) or folded into the scale (which would
    multiply it) is off by orders of magnitude, not by a rounding step.
    """
    shape = (1, 1, 64, 128)
    width = shape[-1]
    torch.manual_seed(11)
    x = torch.randn(shape, dtype=torch.float32)
    w = torch.randn(width, dtype=torch.float32)
    b = torch.randn(width, dtype=torch.float32) * 100.0
    out = ttnn.to_torch(
        rms_norm_ttnn(
            _dev(x, device, dtype=ttnn.float32),
            epsilon=1e-12,
            weight=_dev(w.reshape(1, 1, 1, width), device, dtype=ttnn.float32),
            bias=_dev(b.reshape(1, 1, 1, width), device, dtype=ttnn.float32),
        )
    ).float()
    expected = torch_rms_norm_ttnn(x, epsilon=1e-12, weight=w, bias=b).float()
    # RELATIVE to the output's own scale, deliberately.  The default compute
    # config accumulates in a 16-bit DEST, so a value of magnitude ~100 carries a
    # ~0.4% (bf16) rounding step -- an absolute bound would be measuring that,
    # not the thing under test.  What is under test is the STAGE ORDER, and a
    # wrong order is wrong by ~|bias| (order 100), not by a rounding step.
    scale = expected.abs().max().item()
    worst = (out - expected).abs().max().item()
    assert worst < 0.02 * scale, (
        f"bias-dominant case diverged by {worst} against an output scale of {scale}; "
        f"an error of order |bias| means the bias entered before the normalize "
        f"or was folded into the scale"
    )


def test_single_hot_row_normalizes_to_the_row_length(device):
    """One nonzero element per row: mean(t^2) = 1/W, so that element becomes sqrt(W).

    Hand-calculation for W = 64: the hot lane comes out sqrt(64) = 8 and every
    other lane 0.  This pins the reduce's DIVISOR to the LOGICAL width -- a
    reduce that divided by the padded width would give sqrt(W_padded) instead,
    which is the fault a random input hides inside a uniform scale error.
    """
    width = 64
    x = torch.zeros(1, 1, 32, width, dtype=torch.float32)
    x[..., 0] = 1.0
    out = ttnn.to_torch(rms_norm_ttnn(_dev(x, device, dtype=ttnn.float32), epsilon=1e-12)).float()
    hot = out[..., 0]
    cold = out[..., 1:]
    assert torch.allclose(hot, torch.full_like(hot, width**0.5), rtol=2e-2, atol=2e-2), (
        f"the hot lane must come out sqrt(W) = {width ** 0.5}; got {hot.flatten()[:4].tolist()} "
        f"(a smaller value means the reduce divided by the PADDED width)"
    )
    assert cold.abs().max() < 1e-3, f"cold lanes must stay 0; max {cold.abs().max().item()}"


def test_single_hot_row_non_aligned_width(device):
    """The same pin at W = 50, where the reduce runs on the masked path.

    sqrt(50) != sqrt(64), so this separates "divides by the logical width" from
    "divides by the padded width" by 13% -- far above bf16's own step.
    """
    width = 50
    x = torch.zeros(1, 1, 32, width, dtype=torch.float32)
    x[..., 0] = 1.0
    out = ttnn.to_torch(rms_norm_ttnn(_dev(x, device, dtype=ttnn.float32), epsilon=1e-12)).float()
    hot = out[..., 0]
    assert torch.allclose(hot, torch.full_like(hot, width**0.5), rtol=2e-2, atol=2e-2), (
        f"W = 50 must give sqrt(50) = {width ** 0.5}; got {hot.flatten()[:4].tolist()} "
        f"(sqrt(64) = 8.0 would mean the pad lanes were reduced too)"
    )


def test_zero_scalar_is_zero_not_nan(device):
    """Rank 0, x = 0: epsilon is the entire denominator and 0 * rsqrt(eps) = 0."""
    out = ttnn.to_torch(
        rms_norm_ttnn(
            _dev(torch.zeros((), dtype=torch.float32), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
            epsilon=1e-12,
        )
    )
    assert torch.isfinite(out).all(), "a zero scalar must come out as zero, never NaN"
    torch.testing.assert_close(out.reshape(()), torch.zeros((), dtype=out.dtype), rtol=0, atol=0)


def test_scalar_normalizes_to_its_own_sign(device):
    """Rank 0, x != 0: mean(t^2) = t^2, so t / sqrt(t^2 + eps) = sign(t).

    A one-element row has nothing to average over, which is what makes the
    scalar case a closed form rather than an approximation.
    """
    for value in (3.5, -0.25):
        out = ttnn.to_torch(
            rms_norm_ttnn(
                _dev(
                    torch.tensor(value, dtype=torch.float32),
                    device,
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                epsilon=1e-12,
            )
        ).reshape(())
        assert abs(out.item() - (1.0 if value > 0 else -1.0)) < 2e-2, f"scalar {value} -> {out.item()}, expected sign"
