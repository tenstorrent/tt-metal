# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp


def torch_snake_beta(x, alpha, beta):
    """Reference y = x + sin(alpha*x)^2 / beta, computed in fp32 then cast to x.dtype."""
    out_dtype = x.dtype
    x32 = x.to(torch.float32)
    alpha32 = alpha.to(torch.float32)
    beta32 = beta.to(torch.float32)
    s = torch.sin(alpha32 * x32)
    return (x32 + (s * s) / beta32).to(out_dtype)


_AB_RANGE_ALPHA = (0.96, 1.40)
_AB_RANGE_BETA = (0.97, 1.61)


def _sample_uniform(shape, lo, hi, dtype):
    return (torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo).to(dtype)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)])
@pytest.mark.parametrize(
    "x_shape, ab_shape, seed",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), 1),  # NONE
        ((1, 1, 640, 48), (48,), 0),  # ROW_BCAST
    ],
)
def test_snake_beta_broadcast(device, x_shape, ab_shape, seed, torch_dtype, ttnn_dtype):
    torch.manual_seed(seed)
    x = torch.randn(x_shape, dtype=torch_dtype)
    a = _sample_uniform(ab_shape, *_AB_RANGE_ALPHA, dtype=torch_dtype)
    b = _sample_uniform(ab_shape, *_AB_RANGE_BETA, dtype=torch_dtype)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn_dtype, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))

    if torch_dtype == torch.float32:
        # ULP is a poor metric near y≈0 at fp32 precision; use combined rtol/atol instead.
        assert torch.allclose(expected, result, rtol=1e-5, atol=1e-5)
    else:
        assert_with_ulp(expected, result, ulp_threshold=2)


# Per-preset bf16 distribution params from issue #43337:
# (shape, x_{mean,std,min,max}, a_{mean,std,min,max}, b_{mean,std,min,max}).
_REAL_WORKLOAD_PRESETS = {
    "c1": (
        (1, 1, 2250, 768),
        -0.34375,
        0.486328125,
        -3.265625,
        3.1875,
        1.171875,
        0.0260009765625,
        1.0703125,
        1.3984375,
        1.21875,
        0.033203125,
        1.1171875,
        1.609375,
    ),
    "c2": (
        (1, 1, 36000, 192),
        -0.08935546875,
        0.29296875,
        -2.453125,
        1.8203125,
        1.0703125,
        0.037109375,
        0.96875,
        1.1640625,
        1.078125,
        0.04541015625,
        0.97265625,
        1.203125,
    ),
    "c3": (
        (1, 1, 144000, 48),
        0.01129150390625,
        0.049072265625,
        -0.609375,
        0.76171875,
        1.0625,
        0.05224609375,
        0.95703125,
        1.171875,
        1.2421875,
        0.11083984375,
        1.015625,
        1.5078125,
    ),
}


def _sample_clamped_normal(shape, mean, std, lo, hi, dtype=torch.bfloat16):
    t = torch.randn(shape, dtype=torch.float32) * std + mean
    return t.clamp_(lo, hi).to(dtype)


@pytest.mark.parametrize("torch_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("preset", ["c1", "c2", "c3"])
def test_snake_beta_real_workload(device, preset, torch_dtype):
    """Real-workload distributions from issue #43337; bf16 asserts <= 2 ULP, fp32 uses rtol/atol."""
    torch.manual_seed(0)
    shape, xm, xs, xlo, xhi, am, as_, alo, ahi, bm, bs, blo, bhi = _REAL_WORKLOAD_PRESETS[preset]
    D = shape[-1]

    x = _sample_clamped_normal(shape, xm, xs, xlo, xhi, dtype=torch_dtype)
    a = _sample_clamped_normal((D,), am, as_, alo, ahi, dtype=torch_dtype)
    b = _sample_clamped_normal((D,), bm, bs, blo, bhi, dtype=torch_dtype)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))

    if torch_dtype == torch.float32:
        assert torch.allclose(expected, result, rtol=1e-5, atol=1e-5)
    else:
        assert_with_ulp(expected, result, ulp_threshold=2)


def test_snake_beta_pi_boundary(device):
    """alpha*x at multiples of pi/2 to exercise range-reduction boundaries."""
    import math

    x_vals = torch.tensor(
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, -math.pi / 2, -math.pi],
        dtype=torch.bfloat16,
    )
    x_tile = x_vals.repeat(math.ceil(1024 / len(x_vals)))[:1024].reshape(1, 1, 32, 32)
    a = torch.full((32,), 1.0, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x_tile, a, b)

    x_tt = ttnn.from_torch(x_tile, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected, result, ulp_threshold=1)


@pytest.mark.parametrize(
    "x_shape, ab_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32)),  # NONE: exercises compute_output_specs early return
        ((1, 1, 640, 48), (48,)),  # ROW_BCAST: validates preallocated shape matches broadcast output
    ],
)
def test_snake_beta_output_tensor(device, x_shape, ab_shape):
    """Preallocated output_tensor= path; verifies the result aliases the caller's tensor."""
    torch.manual_seed(0)
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    a = _sample_uniform(ab_shape, *_AB_RANGE_ALPHA, dtype=torch.bfloat16)
    b = _sample_uniform(ab_shape, *_AB_RANGE_BETA, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    out_pre = ttnn.from_torch(torch.zeros(x_shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    ttnn.snake_beta(x_tt, a_tt, b_tt, output_tensor=out_pre)

    # Returned tensor must alias the caller-supplied buffer (proves the op wrote into it
    # instead of allocating a fresh one).
    out_pre_torch = ttnn.to_torch(out_pre)
    assert_with_ulp(expected, out_pre_torch, ulp_threshold=2)


@pytest.mark.parametrize(
    "x_shape, ab_shape",
    [
        ((640, 48), (48,)),  # canonical customer pattern x=(L,D), alpha=beta=(D,)
        ((2, 640, 48), (48,)),  # 3D batched (B,L,D) with vector alpha/beta
    ],
)
def test_snake_beta_rank_preservation(device, x_shape, ab_shape):
    """Output rank must equal input rank for non-rank-4 inputs (issue #43337 caller uses (L,D))."""
    torch.manual_seed(0)
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    a = _sample_uniform(ab_shape, *_AB_RANGE_ALPHA, dtype=torch.bfloat16)
    b = _sample_uniform(ab_shape, *_AB_RANGE_BETA, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.snake_beta(x_tt, a_tt, b_tt)

    assert tuple(result.shape) == x_shape, f"expected output shape {x_shape}, got {tuple(result.shape)}"
    assert_with_ulp(expected, ttnn.to_torch(result), ulp_threshold=2)


# --- Validation failure tests ---


def test_snake_beta_shape_mismatch_alpha_beta(device, expect_error):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, r"alpha.shape == beta.shape|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_dtype_mismatch(device, expect_error):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.float32), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, r"dtype"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_w_mismatch(device, expect_error):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, r"input.W == alpha.W|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_unsupported_broadcast(device, expect_error):
    """alpha with a non-1 non-W dim (H=64) is rejected in v1."""
    x = ttnn.from_torch(torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, r"non-1 size only on the last dim|input.W == alpha.W|broadcast"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_row_major_layout(device, expect_error):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, r"[Tt]ile [Ll]ayout|tile layout"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_unsupported_input_rank(device, expect_error):
    """Rank-1 input is rejected; the kernel reader requires padded_shape rank >= 2."""
    x = ttnn.from_torch(torch.zeros(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, r"input rank >= 2"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_unsupported_dtype_int32(device, expect_error):
    """INT32 is rejected; only BFLOAT16 / FLOAT32 are supported."""
    x = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.int32),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.int32,
        device=device,
    )
    a = ttnn.from_torch(
        torch.ones(32, dtype=torch.int32),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.int32,
        device=device,
    )
    b = ttnn.from_torch(
        torch.ones(32, dtype=torch.int32),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.int32,
        device=device,
    )
    with expect_error(RuntimeError, r"BFLOAT16 or FLOAT32"):
        ttnn.snake_beta(x, a, b)
