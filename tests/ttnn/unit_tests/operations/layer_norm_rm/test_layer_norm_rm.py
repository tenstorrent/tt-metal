# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for layer_norm_rm.

This file is the immutable specification. Do NOT modify.

Tests layer normalization on row-major interleaved tensors with optional
gamma (scale) and beta (shift) affine parameters.

Math:
    output = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

where mean and var are computed per-row (last dimension).
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


def torch_layer_norm_rm(x, gamma=None, beta=None, epsilon=1e-5):
    """PyTorch reference implementation."""
    W = x.shape[-1]
    return F.layer_norm(x, [W], weight=gamma, bias=beta, eps=epsilon)


# ---------------------------------------------------------------------------
# Shape parametrization
# ---------------------------------------------------------------------------

SHAPES_2D = [
    (32, 32),  # single tile
    (64, 128),  # multi-tile, rectangular
    (32, 256),  # single tile-row, wide
    (128, 64),  # tall, narrow
]

SHAPES_3D = [
    (2, 32, 64),  # batched 3D
]

SHAPES_4D = [
    (1, 1, 32, 32),  # single tile, 4D
    (2, 1, 64, 128),  # multi-batch, 4D
    (1, 2, 32, 256),  # multi-channel, wide
    (2, 2, 64, 64),  # multi-batch multi-channel
]

ALL_SHAPES = SHAPES_2D + SHAPES_3D + SHAPES_4D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_device(tensor_torch, device, dtype=ttnn.bfloat16):
    """Convert torch tensor to ttnn on-device tensor in ROW_MAJOR layout."""
    return ttnn.from_torch(
        tensor_torch,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def from_device(tensor_ttnn):
    """Convert ttnn tensor back to torch."""
    return ttnn.to_torch(tensor_ttnn)


# ---------------------------------------------------------------------------
# Test: Pure normalization (no gamma, no beta)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_layer_norm_rm_pure(shape, device):
    """Layer norm without affine parameters."""
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch.float()).bfloat16()

    x_tt = to_device(x_torch, device)
    result_tt = layer_norm_rm(x_tt)
    result_torch = from_device(result_tt)

    torch.testing.assert_close(
        result_torch,
        expected,
        rtol=0.05,
        atol=0.2,
    )


# ---------------------------------------------------------------------------
# Test: With gamma only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_layer_norm_rm_gamma(shape, device):
    """Layer norm with gamma (scale) only."""
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch.float(), gamma=gamma_torch.float()).bfloat16()

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(x_tt, gamma_tt)
    result_torch = from_device(result_tt)

    torch.testing.assert_close(
        result_torch,
        expected,
        rtol=0.05,
        atol=0.2,
    )


# ---------------------------------------------------------------------------
# Test: With gamma and beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_layer_norm_rm_gamma_beta(shape, device):
    """Layer norm with gamma (scale) and beta (shift)."""
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch.float(), gamma=gamma_torch.float(), beta=beta_torch.float()).bfloat16()

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(x_tt, gamma_tt, beta_tt)
    result_torch = from_device(result_tt)

    torch.testing.assert_close(
        result_torch,
        expected,
        rtol=0.05,
        atol=0.2,
    )


# ---------------------------------------------------------------------------
# Test: Custom epsilon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-3, 1e-5, 1e-7])
def test_layer_norm_rm_epsilon(epsilon, device):
    """Layer norm with custom epsilon values."""
    torch.manual_seed(42)
    shape = (64, 128)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(
        x_torch.float(),
        gamma=gamma_torch.float(),
        beta=beta_torch.float(),
        epsilon=epsilon,
    ).bfloat16()

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(x_tt, gamma_tt, beta_tt, epsilon=epsilon)
    result_torch = from_device(result_tt)

    torch.testing.assert_close(
        result_torch,
        expected,
        rtol=0.05,
        atol=0.2,
    )


# ---------------------------------------------------------------------------
# Test: Beta only (no gamma)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (64, 128)])
def test_layer_norm_rm_beta_only(shape, device):
    """Layer norm with beta (shift) only, no gamma."""
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch.float(), beta=beta_torch.float()).bfloat16()

    x_tt = to_device(x_torch, device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(x_tt, None, beta_tt)
    result_torch = from_device(result_tt)

    torch.testing.assert_close(
        result_torch,
        expected,
        rtol=0.05,
        atol=0.2,
    )


# ---------------------------------------------------------------------------
# Test: Input validation
# ---------------------------------------------------------------------------


def test_layer_norm_rm_rejects_tile_layout(device):
    """Should reject TILE_LAYOUT input."""
    x_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_layer_norm_rm_rejects_wrong_dtype(device):
    """Should reject non-bfloat16 input dtype."""
    x_torch = torch.randn(32, 32, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_layer_norm_rm_rejects_gamma_width_mismatch(device):
    """Should reject gamma with wrong width."""
    torch.manual_seed(42)
    x_torch = torch.randn(32, 64, dtype=torch.bfloat16)
    gamma_torch = torch.randn(32, dtype=torch.bfloat16)  # wrong: 32 != 64

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, 32), device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt, gamma_tt)


# ---------------------------------------------------------------------------
# Test: Output shape and layout preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 64), (2, 1, 64, 128)])
def test_layer_norm_rm_output_properties(shape, device):
    """Output should have same shape, dtype, and layout as input."""
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    x_tt = to_device(x_torch, device)
    result_tt = layer_norm_rm(x_tt)

    assert list(result_tt.shape) == list(x_tt.shape)
    assert result_tt.dtype == x_tt.dtype
    assert result_tt.layout == ttnn.ROW_MAJOR_LAYOUT
