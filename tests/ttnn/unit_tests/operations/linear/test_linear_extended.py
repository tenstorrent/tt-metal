# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for ttnn.operations.linear.linear (Phase 0).

Focused additional coverage beyond test_linear.py. Intentionally small
(verification, not exhaustive sweep — exhaustive coverage belongs in
refinement tests):

  - memory_config kwarg honored (default and explicit DRAM).
  - Output dtype/layout/shape contract.
  - Bias-as-zero degenerates to plain matmul (sanity).
  - Bias-only-rows-1..31 must NOT influence output (only row 0 is the bias).
  - Validation rejects bias height != 32.
  - Single multi-tile shape with both bias modes runs back-to-back without
    state leak.
"""

import pytest
import torch

import ttnn
from ttnn.operations.linear import linear


torch.manual_seed(7)


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _ref(x, w, b=None):
    out = x @ w
    if b is not None:
        out = out + b[..., 0:1, :]
    return out


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64)])
def test_output_contract(device, shape):
    M, K, N = shape
    x = _to_device(torch.randn((1, 1, M, K), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, K, N), dtype=torch.bfloat16), device)
    y = linear(x, w)
    assert list(y.shape) == [1, 1, M, N]
    assert y.dtype == ttnn.bfloat16
    assert y.layout == ttnn.TILE_LAYOUT


# ---------------------------------------------------------------------------
# memory_config kwarg
# ---------------------------------------------------------------------------


def test_memory_config_default_dram(device):
    M, K, N = 64, 64, 64
    x = _to_device(torch.randn((1, 1, M, K), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, K, N), dtype=torch.bfloat16), device)
    y = linear(x, w)
    # Default memory config should be DRAM-interleaved.
    assert y.memory_config().buffer_type == ttnn.BufferType.DRAM


def test_memory_config_explicit_dram(device):
    M, K, N = 64, 64, 64
    x = _to_device(torch.randn((1, 1, M, K), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, K, N), dtype=torch.bfloat16), device)
    y = linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert y.memory_config().buffer_type == ttnn.BufferType.DRAM


# ---------------------------------------------------------------------------
# Bias semantics
# ---------------------------------------------------------------------------


def test_zero_bias_equivalent_to_no_bias(device):
    """linear(x, w, bias=zeros) == linear(x, w) (within bf16 noise)."""
    M, K, N = 64, 64, 64
    x_t = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_t = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    b_t = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)

    x = _to_device(x_t, device)
    w = _to_device(w_t, device)
    b = _to_device(b_t, device)

    y_no_bias = ttnn.to_torch(linear(x, w))
    y_zero_bias = ttnn.to_torch(linear(x, w, bias=b))

    # Allow a very tight tolerance — values pass through the same matmul plus
    # an add-with-zero. Pack/unpack rounding can shift 1 bf16 ULP at most.
    assert torch.allclose(y_no_bias.float(), y_zero_bias.float(), rtol=0.0, atol=1e-2)


def test_bias_rows_1_to_31_are_ignored(device):
    """Only row 0 of the [1, 1, 32, N] bias tile is consumed by the helper.

    Filling rows 1..31 with junk must not change the output.
    """
    M, K, N = 64, 64, 64
    x_t = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_t = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

    bias_row0 = torch.randn(N, dtype=torch.bfloat16)

    b_clean = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)
    b_clean[..., 0, :] = bias_row0

    b_dirty = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)
    b_dirty[..., 0, :] = bias_row0
    b_dirty[..., 1:, :] = torch.randn((31, N), dtype=torch.bfloat16) * 100.0  # large junk

    x = _to_device(x_t, device)
    w = _to_device(w_t, device)
    bc = _to_device(b_clean, device)
    bd = _to_device(b_dirty, device)

    y_clean = ttnn.to_torch(linear(x, w, bias=bc))
    y_dirty = ttnn.to_torch(linear(x, w, bias=bd))

    # Outputs must be identical (RowBroadcast pulls only row 0).
    assert torch.allclose(
        y_clean.float(), y_dirty.float(), rtol=0.0, atol=1e-3
    ), "RowBroadcast bias should ignore rows 1..31; output changed when junk was added"


def test_validate_bias_height_must_be_32(device):
    """bias.shape[-2] != 32 must raise ValueError before any device work."""
    M, K, N = 64, 64, 64
    x = _to_device(torch.randn((1, 1, M, K), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((1, 1, K, N), dtype=torch.bfloat16), device)
    # Build a bias with height 64 (still tile-aligned, but disallowed).
    b_bad = _to_device(torch.zeros((1, 1, 64, N), dtype=torch.bfloat16), device)
    with pytest.raises((ValueError, RuntimeError)):
        linear(x, w, bias=b_bad)


# ---------------------------------------------------------------------------
# State leak between back-to-back calls
# ---------------------------------------------------------------------------


def test_back_to_back_bias_then_no_bias(device):
    """Run a bias call followed by a no-bias call; both must remain correct."""
    M, K, N = 64, 64, 64
    x_t = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    w_t = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    b_t = torch.zeros((1, 1, 32, N), dtype=torch.bfloat16)
    b_t[..., 0, :] = torch.randn(N, dtype=torch.bfloat16)

    x = _to_device(x_t, device)
    w = _to_device(w_t, device)
    b = _to_device(b_t, device)

    y_bias = ttnn.to_torch(linear(x, w, bias=b)).float()
    y_plain = ttnn.to_torch(linear(x, w)).float()

    expected_bias = _ref(x_t, w_t, b_t).float()
    expected_plain = _ref(x_t, w_t, None).float()

    rtol, atol = 0.02, 0.15
    assert torch.allclose(y_bias, expected_bias, rtol=rtol, atol=atol)
    assert torch.allclose(y_plain, expected_plain, rtol=rtol, atol=atol)
