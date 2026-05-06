# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for ttnn.operations.linear.linear (Phase 0).

This test is the spec for the operation: do NOT modify it during implementation.

Coverage:
  - Forward correctness, no bias, multiple shapes (single-tile, square multi-tile,
    non-square, "wide M / tall K" style asymmetries).
  - Forward correctness, with bias, the same shape sweep.
  - Python-side validation (dtype, layout, rank, K mismatch, N mismatch, alignment).

Run from repo root:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/linear/test_linear.py
"""

import pytest
import torch

import ttnn
from ttnn.operations.linear import linear


# ---------------------------------------------------------------------------
# Reference + helpers
# ---------------------------------------------------------------------------

torch.manual_seed(42)


def _reference(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """PyTorch reference for `linear(x, w, bias=b)`.

    x: [1, 1, M, K], w: [1, 1, K, N], b: [1, 1, 32, N] with values in row 0.
    Output: [1, 1, M, N].
    """
    out = x @ w
    if b is not None:
        # Bias values live in row 0 of the [1, 1, 32, N] tile-padded layout.
        # Broadcast that single row across all M rows.
        out = out + b[..., 0:1, :]  # shape [1, 1, 1, N], broadcasts on M
    return out


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_bias_tensor(N: int, device) -> tuple[torch.Tensor, ttnn.Tensor]:
    """Build a [1, 1, 32, N] bias tensor: row 0 random, rows 1..31 zero."""
    b_torch = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)
    b_torch[..., 0, :] = torch.randn(N, dtype=torch.bfloat16)
    return b_torch, _to_device(b_torch, device)


# ---------------------------------------------------------------------------
# Forward correctness — no bias
# ---------------------------------------------------------------------------

# Shape sweep: (M, K, N), each divisible by 32, exercising single-tile, square
# multi-tile, non-square wide-N, non-square tall-K, and a mixed case.
_FORWARD_SHAPES = [
    pytest.param((32, 32, 32), id="single_tile_M32_K32_N32"),
    pytest.param((64, 64, 64), id="square_M64_K64_N64"),
    pytest.param((32, 128, 64), id="tall_K_M32_K128_N64"),
    pytest.param((96, 32, 128), id="wide_N_M96_K32_N128"),
    pytest.param((128, 96, 64), id="mixed_M128_K96_N64"),
]


@pytest.mark.parametrize("shape", _FORWARD_SHAPES)
def test_linear_no_bias(device, shape):
    M, K, N = shape

    x_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

    x = _to_device(x_torch, device)
    w = _to_device(w_torch, device)

    y = linear(x, w)

    assert list(y.shape) == [1, 1, M, N], f"output shape {y.shape} != [1,1,{M},{N}]"
    assert y.dtype == ttnn.bfloat16
    assert y.layout == ttnn.TILE_LAYOUT

    y_torch = ttnn.to_torch(y)
    expected = _reference(x_torch, w_torch, None)

    # Matmul accumulates K terms; tolerances scale with K.
    rtol, atol = 0.02, 0.1
    assert torch.allclose(y_torch.float(), expected.float(), rtol=rtol, atol=atol), (
        f"linear(x, w) mismatch for shape {shape}; " f"max abs diff = {(y_torch - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Forward correctness — with bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", _FORWARD_SHAPES)
def test_linear_with_bias(device, shape):
    M, K, N = shape

    x_torch = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_torch = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    b_torch, b = _make_bias_tensor(N, device)

    x = _to_device(x_torch, device)
    w = _to_device(w_torch, device)

    y = linear(x, w, bias=b)

    assert list(y.shape) == [1, 1, M, N]
    assert y.dtype == ttnn.bfloat16
    assert y.layout == ttnn.TILE_LAYOUT

    y_torch = ttnn.to_torch(y)
    expected = _reference(x_torch, w_torch, b_torch)

    # Bias path adds one more rounding step on top of matmul; loosen atol slightly.
    rtol, atol = 0.02, 0.15
    assert torch.allclose(y_torch.float(), expected.float(), rtol=rtol, atol=atol), (
        f"linear(x, w, bias=b) mismatch for shape {shape}; " f"max abs diff = {(y_torch - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Python-side validation
#
# linear() must reject malformed inputs BEFORE allocating an output tensor or
# building a program descriptor. ValueError is the contracted exception type;
# RuntimeError is also accepted as a fallback.
# ---------------------------------------------------------------------------


def _good_inputs(device, M=64, K=64, N=64):
    x = _to_device(torch.randn((1, 1, M, K), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, K, N), dtype=torch.bfloat16), device)
    return x, w


def test_validate_input_dtype_must_be_bfloat16(device):
    M, K, N = 64, 64, 64
    x_bad = ttnn.from_torch(
        torch.randn((1, 1, M, K), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _, w = _good_inputs(device, M, K, N)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x_bad, w)


def test_validate_weight_dtype_must_be_bfloat16(device):
    M, K, N = 64, 64, 64
    x, _ = _good_inputs(device, M, K, N)
    w_bad = ttnn.from_torch(
        torch.randn((1, 1, K, N), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises((ValueError, RuntimeError)):
        linear(x, w_bad)


def test_validate_layout_must_be_tile(device):
    M, K, N = 64, 64, 64
    x_rm = ttnn.from_torch(
        torch.randn((1, 1, M, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _, w = _good_inputs(device, M, K, N)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x_rm, w)


def test_validate_rank_must_be_4(device):
    M, K, N = 64, 64, 64
    x_2d = ttnn.from_torch(
        torch.randn((M, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _, w = _good_inputs(device, M, K, N)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x_2d, w)


def test_validate_inner_dim_mismatch(device):
    # input K = 64, weight K = 96 — should be rejected.
    x = _to_device(torch.randn((1, 1, 64, 64), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, 96, 64), dtype=torch.bfloat16), device)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x, w)


def test_validate_bias_n_mismatch(device):
    M, K, N = 64, 64, 64
    x, w = _good_inputs(device, M, K, N)
    # bias has N=128 but weight has N=64.
    b_bad = _to_device(torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16), device)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x, w, bias=b_bad)


def test_validate_M_not_tile_aligned(device):
    # M=33 not divisible by 32. ttnn.from_torch may pad internally, so we feed
    # a "raw" tensor whose padded shape still encodes the bad logical size.
    # Simplest path: build a tile-aligned tensor and rely on linear() inspecting
    # logical shape. Here we use the closest unaligned logical shape achievable
    # by passing a [1, 1, 33, 64] torch tensor.
    bad = torch.randn((1, 1, 33, 64), dtype=torch.bfloat16)
    try:
        x_bad = ttnn.from_torch(
            bad,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception:
        # If TTNN itself rejects the shape during from_torch, the contract still
        # holds (the user can never reach linear() with a bad shape).
        pytest.skip("ttnn.from_torch rejected the unaligned shape upstream")
    w = _to_device(torch.randn((1, 1, 64, 64), dtype=torch.bfloat16), device)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x_bad, w)


def test_validate_K_not_tile_aligned(device):
    bad_x = torch.randn((1, 1, 64, 33), dtype=torch.bfloat16)
    bad_w = torch.randn((1, 1, 33, 64), dtype=torch.bfloat16)
    try:
        x_bad = ttnn.from_torch(
            bad_x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w_bad = ttnn.from_torch(
            bad_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception:
        pytest.skip("ttnn.from_torch rejected the unaligned shape upstream")
    with pytest.raises((ValueError, RuntimeError)):
        linear(x_bad, w_bad)


def test_validate_N_not_tile_aligned(device):
    bad_w = torch.randn((1, 1, 64, 33), dtype=torch.bfloat16)
    x = _to_device(torch.randn((1, 1, 64, 64), dtype=torch.bfloat16), device)
    try:
        w_bad = ttnn.from_torch(
            bad_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception:
        pytest.skip("ttnn.from_torch rejected the unaligned shape upstream")
    with pytest.raises((ValueError, RuntimeError)):
        linear(x, w_bad)
