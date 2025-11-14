# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
import numpy as np


# Test shape parameters
SHAPES_SMALL = [
    (1, 1, 32, 32),
    (2, 1, 32, 64),
    (4, 1, 16, 32),
    (1, 1, 64, 32),
    (2, 1, 16, 64),
]

SHAPES_MEDIUM = [
    (4, 1, 64, 64),
    (8, 1, 32, 64),
    (2, 1, 128, 32),
    (4, 1, 32, 128),
    (1, 1, 128, 64),
]

SHAPES_LARGE = [
    (4, 1, 128, 128),
    (8, 1, 64, 128),
    (2, 1, 256, 64),
    (4, 1, 64, 256),
    (1, 1, 256, 128),
]

SHAPES_VERY_LARGE = [
    (8, 1, 128, 256),
    (4, 1, 256, 256),
    (2, 1, 512, 128),
    (1, 1, 512, 256),
]

SHAPES_EXTRA_LARGE = [
    # Large batch sizes
    (16, 1, 128, 128),
    (32, 1, 64, 128),
    (8, 1, 256, 256),
    # Large spatial dimensions
    (4, 1, 512, 512),
    (2, 1, 1024, 256),
    (1, 1, 1024, 512),
    # Mixed large dimensions
    (8, 1, 256, 512),
    (4, 1, 512, 256),
    (16, 1, 128, 256),
]

ALL_SHAPES = SHAPES_SMALL + SHAPES_MEDIUM + SHAPES_LARGE + SHAPES_VERY_LARGE


def test_clip_grad_norm_below_max(device):
    """Test that gradients are not scaled when norm is below max_norm."""
    # Create a tensor with small values (norm will be below max_norm)
    t = (
        torch.ones(
            (
                4,
                1,
                32,
                64,
            )
        )
        * 0.5
    )  # Small values to keep norm low
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    max_norm = 10.0  # Large max_norm, so no clipping should occur
    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=1e-12)
    output_torch = ttnn.to_torch(output)

    # When norm is below max_norm, output should equal input (scale = 1.0)
    # Convert input to bfloat16 for comparison
    t_bfloat16 = t.to(torch.bfloat16)
    assert torch.allclose(output_torch, t_bfloat16, rtol=0.1, atol=0.1)


def test_clip_grad_norm_above_max(device):
    """Test that gradients are scaled when norm exceeds max_norm."""
    # Create a tensor with large values (norm will exceed max_norm)
    t = (
        torch.ones(
            (
                4,
                1,
                32,
                64,
            )
        )
        * 5.0
    )  # Large values to make norm exceed max_norm
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    max_norm = 2.0  # Small max_norm, so clipping should occur
    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=1e-12)
    output_torch = ttnn.to_torch(output)

    # Compute expected norm and scale
    flat_t = t.reshape(-1)
    expected_norm = torch.norm(flat_t, p=2.0).item()
    expected_scale = max_norm / (expected_norm + 1e-12)
    expected_output = t * expected_scale

    # Output should be scaled down
    assert torch.allclose(output_torch, expected_output, rtol=0.2, atol=0.2)
    # Verify that the output norm is approximately max_norm
    output_norm = torch.norm(output_torch.reshape(-1), p=2.0).item()
    assert np.allclose(output_norm, max_norm, rtol=0.2)


def test_clip_grad_norm_l2_norm(device):
    """Test clip_grad_norm with L2 norm (p=2.0)."""
    t = (
        torch.randn(
            (
                2,
                1,
                32,
                32,
            )
        )
        * 3.0
    )  # Random values
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    max_norm = 5.0
    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=1e-12)
    output_torch = ttnn.to_torch(output)

    # Compute reference
    flat_t = t.reshape(-1)
    ref_norm = torch.norm(flat_t, p=2.0).item()
    ref_scale = 1.0 if ref_norm <= max_norm else (max_norm / (ref_norm + 1e-12))
    ref_output = t * ref_scale

    # Verify output matches reference
    assert torch.allclose(output_torch, ref_output, rtol=0.2, atol=0.2)


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_clip_grad_norm_different_shapes(device, shape):
    """Test clip_grad_norm with different tensor shapes."""
    max_norm = 3.0
    t = torch.randn(shape) * 2.0
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=1e-12)
    output_torch = ttnn.to_torch(output)

    # Compute reference
    flat_t = t.reshape(-1)
    ref_norm = torch.norm(flat_t, p=2.0).item()
    ref_scale = 1.0 if ref_norm <= max_norm else (max_norm / (ref_norm + 1e-12))
    ref_output = t * ref_scale

    # Verify output matches reference
    assert torch.allclose(output_torch, ref_output, rtol=0.2, atol=0.2), f"Failed for shape {shape}"


@pytest.mark.parametrize("shape", SHAPES_EXTRA_LARGE)
def test_clip_grad_norm_large_shapes(device, shape):
    """Test clip_grad_norm with large tensor shapes."""
    max_norm = 5.0
    t = torch.randn(shape) * 1.5
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=1e-12)
    output_torch = ttnn.to_torch(output)

    # Compute reference
    flat_t = t.reshape(-1)
    ref_norm = torch.norm(flat_t, p=2.0).item()
    ref_scale = 1.0 if ref_norm <= max_norm else (max_norm / (ref_norm + 1e-12))
    ref_output = t * ref_scale

    # Verify output matches reference
    assert torch.allclose(output_torch, ref_output, rtol=0.2, atol=0.2), f"Failed for large shape {shape}"

    # Verify norm constraint
    output_norm = torch.norm(output_torch.reshape(-1), p=2.0).item()
    if ref_norm > max_norm:
        # If input was clipped, output norm should be close to max_norm
        assert np.allclose(
            output_norm, max_norm, rtol=0.3
        ), f"Output norm {output_norm} not close to max_norm {max_norm} for shape {shape}"


def test_clip_grad_norm_eps(device):
    """Test that eps parameter is used correctly."""
    t = (
        torch.ones(
            (
                2,
                1,
                32,
                32,
            )
        )
        * 10.0
    )  # Large values
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    max_norm = 5.0
    eps = 1e-6
    output = ttnn.experimental.clip_grad_norm(t_tt, max_norm=max_norm, p=2.0, eps=eps)
    output_torch = ttnn.to_torch(output)

    # Compute reference with eps
    flat_t = t.reshape(-1)
    ref_norm = torch.norm(flat_t, p=2.0).item()
    ref_scale = max_norm / (ref_norm + eps)
    ref_output = t * ref_scale

    # Verify output matches reference
    assert torch.allclose(output_torch, ref_output, rtol=0.2, atol=0.2)
