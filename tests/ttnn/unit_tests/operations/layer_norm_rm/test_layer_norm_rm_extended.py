# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for layer_norm_rm.

Broad coverage across shapes, ranks, parameters, and optional inputs.
Includes capability probes for documenting boundaries.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def torch_layer_norm_rm(x, gamma=None, beta=None, epsilon=1e-5):
    """PyTorch reference computed in float32."""
    W = x.shape[-1]
    x_f32 = x.float()
    g = gamma.float() if gamma is not None else None
    b = beta.float() if beta is not None else None
    return F.layer_norm(x_f32, [W], weight=g, bias=b, eps=epsilon).to(x.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_device(tensor_torch, device):
    return ttnn.from_torch(
        tensor_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def assert_close(result_torch, expected, pcc_min=0.999, atol=0.2, rtol=0.05):
    """Combined PCC + allclose assertion."""
    a = result_torch.float().flatten()
    e = expected.float().flatten()
    a_c = a - a.mean()
    e_c = e - e.mean()
    num = (a_c * e_c).sum()
    den = a_c.norm() * e_c.norm()
    pcc = (num / den).item() if den > 1e-30 else 1.0
    assert pcc >= pcc_min, f"PCC too low: {pcc:.6f} < {pcc_min}"
    torch.testing.assert_close(result_torch, expected, rtol=rtol, atol=atol)


# ===========================================================================
# 1. Shape coverage — 2D
# ===========================================================================

SHAPES_2D = [
    pytest.param((32, 32), id="2d_32x32"),
    pytest.param((32, 64), id="2d_32x64"),
    pytest.param((64, 32), id="2d_64x32"),
    pytest.param((64, 128), id="2d_64x128"),
    pytest.param((128, 64), id="2d_128x64"),
    pytest.param((128, 256), id="2d_128x256"),
    pytest.param((256, 32), id="2d_256x32"),
    pytest.param((32, 512), id="2d_32x512"),
    pytest.param((32, 768), id="2d_32x768"),
]


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_2d_shapes_pure(shape, device):
    """2D shapes without affine."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x)
    result = ttnn.to_torch(layer_norm_rm(to_device(x, device)))
    assert_close(result, expected)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_2d_shapes_gamma_beta(shape, device):
    """2D shapes with gamma+beta."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


# ===========================================================================
# 2. Shape coverage — 3D
# ===========================================================================

SHAPES_3D = [
    pytest.param((2, 32, 64), id="3d_2x32x64"),
    pytest.param((3, 64, 128), id="3d_3x64x128"),
    pytest.param((4, 32, 32), id="3d_4x32x32"),
    pytest.param((1, 128, 256), id="3d_1x128x256"),
]


@pytest.mark.parametrize("shape", SHAPES_3D)
def test_3d_shapes_gamma_beta(shape, device):
    """3D shapes with gamma+beta."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


# ===========================================================================
# 3. Shape coverage — 4D
# ===========================================================================

SHAPES_4D = [
    pytest.param((1, 1, 32, 32), id="4d_1x1x32x32"),
    pytest.param((2, 1, 64, 128), id="4d_2x1x64x128"),
    pytest.param((1, 2, 32, 256), id="4d_1x2x32x256"),
    pytest.param((2, 2, 64, 64), id="4d_2x2x64x64"),
    pytest.param((4, 2, 32, 128), id="4d_4x2x32x128"),
    pytest.param((8, 1, 32, 64), id="4d_8x1x32x64"),
    pytest.param((1, 8, 64, 32), id="4d_1x8x64x32"),
    pytest.param((4, 4, 32, 32), id="4d_4x4x32x32"),
]


@pytest.mark.parametrize("shape", SHAPES_4D)
def test_4d_shapes_gamma_beta(shape, device):
    """4D shapes with gamma+beta."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


# ===========================================================================
# 4. Affine mode cross-product
# ===========================================================================

AFFINE_SHAPES = [
    pytest.param((32, 64), id="32x64"),
    pytest.param((64, 128), id="64x128"),
    pytest.param((2, 1, 32, 256), id="2x1x32x256"),
]


@pytest.mark.parametrize("shape", AFFINE_SHAPES)
def test_gamma_only(shape, device):
    """Gamma only, no beta."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


@pytest.mark.parametrize("shape", AFFINE_SHAPES)
def test_beta_only(shape, device):
    """Beta only, no gamma."""
    torch.manual_seed(42)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            None,
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


@pytest.mark.parametrize("shape", AFFINE_SHAPES)
def test_no_affine(shape, device):
    """No gamma, no beta (pure normalization)."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x)
    result = ttnn.to_torch(layer_norm_rm(to_device(x, device)))
    assert_close(result, expected)


# ===========================================================================
# 5. Epsilon variations
# ===========================================================================

EPSILON_VALUES = [
    pytest.param(1e-7, id="eps_1e-7"),
    pytest.param(1e-6, id="eps_1e-6"),
    pytest.param(1e-5, id="eps_1e-5_default"),
    pytest.param(1e-3, id="eps_1e-3"),
    pytest.param(1e-2, id="eps_1e-2"),
    pytest.param(0.1, id="eps_0.1"),
]


@pytest.mark.parametrize("epsilon", EPSILON_VALUES)
def test_epsilon_values(epsilon, device):
    """Various epsilon values."""
    torch.manual_seed(42)
    shape = (64, 128)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b, epsilon=epsilon)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
            epsilon=epsilon,
        )
    )
    assert_close(result, expected)


# ===========================================================================
# 6. Data distributions
# ===========================================================================


def test_uniform_distribution(device):
    """Uniform [0, 1] input."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    W = shape[-1]
    x = torch.rand(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


def test_small_magnitude(device):
    """Small magnitude input (near zero)."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    W = shape[-1]
    x = (torch.randn(shape) * 0.01).to(torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    # Slightly relaxed for small inputs
    assert_close(result, expected, pcc_min=0.998, atol=0.3, rtol=0.1)


def test_large_magnitude(device):
    """Large magnitude input."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    W = shape[-1]
    x = (torch.randn(shape) * 10.0).to(torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


def test_identity_affine(device):
    """gamma=1, beta=0 should match pure normalization."""
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.ones(W, dtype=torch.bfloat16)
    b = torch.zeros(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


# ===========================================================================
# 7. Input validation
# ===========================================================================


def test_rejects_1d_input(device):
    """Should reject 1D input (rank < 2)."""
    x = torch.randn(32, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_rejects_non_tile_aligned_width(device):
    """Should reject width not multiple of 32."""
    x = torch.randn(32, 50, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_rejects_non_tile_aligned_height(device):
    """Should reject height not multiple of 32."""
    x = torch.randn(50, 32, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_rejects_tile_layout(device):
    """Should reject TILE_LAYOUT input."""
    x = torch.randn(32, 32, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_rejects_float32(device):
    """Should reject float32 input."""
    x = torch.randn(32, 32, dtype=torch.float32)
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(x_tt)


def test_rejects_gamma_width_mismatch(device):
    """Gamma width must match input width."""
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    g = torch.randn(32, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, 32), device),
        )


def test_rejects_beta_width_mismatch(device):
    """Beta width must match input width."""
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    g = torch.randn(64, dtype=torch.bfloat16)
    b = torch.randn(32, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, 64), device),
            to_device(b.reshape(1, 1, 1, 32), device),
        )


# ===========================================================================
# 8. Output properties
# ===========================================================================

OUTPUT_SHAPES = [
    pytest.param((32, 64), id="2d_32x64"),
    pytest.param((2, 32, 64), id="3d_2x32x64"),
    pytest.param((1, 2, 64, 128), id="4d_1x2x64x128"),
]


@pytest.mark.parametrize("shape", OUTPUT_SHAPES)
def test_output_shape_dtype_layout(shape, device):
    """Output should preserve shape, dtype, layout."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x_tt = to_device(x, device)
    result_tt = layer_norm_rm(x_tt)
    assert list(result_tt.shape) == list(x_tt.shape)
    assert result_tt.dtype == ttnn.bfloat16
    assert result_tt.layout == ttnn.ROW_MAJOR_LAYOUT


# ===========================================================================
# 9. Capability probes (document boundaries, do not delete)
# ===========================================================================


def test_probe_max_width_512(device):
    """Probe: W=512 should work with gamma+beta."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 512)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


def test_probe_max_width_768(device):
    """Probe: W=768 should work with gamma+beta."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 768)
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(W, dtype=torch.bfloat16)
    b = torch.randn(W, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x, gamma=g, beta=b)
    result = ttnn.to_torch(
        layer_norm_rm(
            to_device(x, device),
            to_device(g.reshape(1, 1, 1, W), device),
            to_device(b.reshape(1, 1, 1, W), device),
        )
    )
    assert_close(result, expected)


def test_probe_large_batch(device):
    """Probe: large batch count (NC=16) with small tiles."""
    torch.manual_seed(42)
    shape = (4, 4, 32, 32)
    x = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_layer_norm_rm(x)
    result = ttnn.to_torch(layer_norm_rm(to_device(x, device)))
    assert_close(result, expected)
