# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the layernorm_fused_rm operation.

This operation performs LayerNorm on row-major input and produces row-major output,
internally performing tilize/untilize to enable tiled compute.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_layernorm(x, gamma, beta, eps):
    """Reference PyTorch implementation of LayerNorm."""
    return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)


# =============================================================================
# Functional Tests
# =============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # Single tile
        (1, 1, 64, 64),  # 2x2 tiles
        (1, 1, 128, 128),  # 4x4 tiles
    ],
    ids=["single_tile", "2x2_tiles", "4x4_tiles"],
)
def test_layernorm_fused_rm_basic_shapes(device, shape):
    """Test layernorm_fused_rm with various tile-aligned shapes."""
    torch.manual_seed(42)

    # Create input tensor
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    # Create gamma and beta (1D tensors matching width)
    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    # Compute reference
    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    # Convert to TTNN tensors (ROW_MAJOR layout)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)

    # Convert back to torch
    output_torch = ttnn.to_torch(output_tensor)

    # Verify output shape matches input shape
    assert (
        output_torch.shape == torch_input.shape
    ), f"Output shape {output_torch.shape} != input shape {torch_input.shape}"

    # Verify layout is ROW_MAJOR
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, f"Output layout should be ROW_MAJOR"

    # Verify numerical accuracy
    assert_with_pcc(torch_output, output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 256),  # Wide: 8 tiles across
        (1, 1, 32, 512),  # Wide: 16 tiles across
        (1, 1, 32, 1024),  # Wide: 32 tiles across
    ],
    ids=["wide_8tiles", "wide_16tiles", "wide_32tiles"],
)
def test_layernorm_fused_rm_wide_tensors(device, shape):
    """Test layernorm_fused_rm with wide tensors (many tiles per row)."""
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert output_torch.shape == torch_input.shape
    assert_with_pcc(torch_output, output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 64, 32),  # Tall: 2 tile rows
        (1, 1, 128, 32),  # Tall: 4 tile rows
        (1, 1, 256, 32),  # Tall: 8 tile rows
    ],
    ids=["tall_2rows", "tall_4rows", "tall_8rows"],
)
def test_layernorm_fused_rm_tall_tensors(device, shape):
    """Test layernorm_fused_rm with tall tensors (many tile rows)."""
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert output_torch.shape == torch_input.shape
    assert_with_pcc(torch_output, output_torch, pcc=0.999)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-12])
def test_layernorm_fused_rm_epsilon_values(device, epsilon):
    """Test layernorm_fused_rm with different epsilon values."""
    torch.manual_seed(42)

    shape = (1, 1, 32, 64)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_torch, pcc=0.999)


def test_layernorm_fused_rm_uniform_input(device):
    """Test layernorm with uniform input (variance=0 case)."""
    torch.manual_seed(42)

    shape = (1, 1, 32, 64)
    width = shape[-1]

    # Uniform input: all same value per row -> variance = 0
    torch_input = torch.full(shape, 5.0, dtype=torch.bfloat16)
    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    # For uniform input, x - mean = 0, so output should be beta
    assert_with_pcc(torch_output, output_torch, pcc=0.99)


def test_layernorm_fused_rm_gamma_one_beta_zero(device):
    """Test layernorm with gamma=1, beta=0 (pure normalization)."""
    torch.manual_seed(42)

    shape = (1, 1, 64, 64)
    width = shape[-1]

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones((width,), dtype=torch.bfloat16)
    torch_beta = torch.zeros((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "batch_shape",
    [
        pytest.param(
            (2, 1, 64, 64),
            marks=pytest.mark.xfail(reason="Batching not fully supported yet - only inner (H,W) dimensions processed"),
        ),
        pytest.param(
            (1, 4, 32, 64),
            marks=pytest.mark.xfail(reason="Batching not fully supported yet - only inner (H,W) dimensions processed"),
        ),
        pytest.param(
            (2, 4, 32, 32),
            marks=pytest.mark.xfail(reason="Batching not fully supported yet - only inner (H,W) dimensions processed"),
        ),
    ],
    ids=["batch_2", "channels_4", "batch_2_channels_4"],
)
def test_layernorm_fused_rm_batched(device, batch_shape):
    """Test layernorm_fused_rm with batched inputs."""
    torch.manual_seed(42)

    torch_input = torch.rand(batch_shape, dtype=torch.bfloat16)
    width = batch_shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert output_torch.shape == torch_input.shape
    assert_with_pcc(torch_output, output_torch, pcc=0.999)


# =============================================================================
# Validation Tests (Error Cases)
# =============================================================================


def test_layernorm_fused_rm_wrong_layout(device):
    """Test that TILE_LAYOUT input raises an error."""
    torch.manual_seed(42)

    shape = (1, 1, 32, 32)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    # Create input with TILE_LAYOUT (should fail)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,  # Wrong layout
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(Exception) as excinfo:
        ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor)

    assert "ROW_MAJOR" in str(excinfo.value).upper() or "layout" in str(excinfo.value).lower()


def test_layernorm_fused_rm_non_tile_aligned_width(device):
    """Test that non-tile-aligned width raises an error."""
    torch.manual_seed(42)

    # Width 48 is not divisible by 32
    shape = (1, 1, 32, 48)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(Exception) as excinfo:
        ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor)

    # Should mention width alignment or multiple of 32
    error_msg = str(excinfo.value).lower()
    assert "width" in error_msg or "32" in error_msg or "align" in error_msg or "multiple" in error_msg


def test_layernorm_fused_rm_non_tile_aligned_height(device):
    """Test that non-tile-aligned height raises an error."""
    torch.manual_seed(42)

    # Height 48 is not divisible by 32
    shape = (1, 1, 48, 32)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(Exception) as excinfo:
        ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor)

    # Should mention height alignment or multiple of 32
    error_msg = str(excinfo.value).lower()
    assert "height" in error_msg or "32" in error_msg or "align" in error_msg or "multiple" in error_msg


def test_layernorm_fused_rm_2d_tensor(device):
    """Test that 2D tensors work (minimum rank is 2)."""
    torch.manual_seed(42)

    # 2D tensor is valid per spec
    shape = (32, 64)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    width = shape[-1]

    torch_gamma = torch.rand((width,), dtype=torch.bfloat16)
    torch_beta = torch.rand((width,), dtype=torch.bfloat16)

    epsilon = 1e-5
    torch_output = torch_layernorm(torch_input, torch_gamma, torch_beta, epsilon)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_tensor = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_tensor = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.layernorm_fused_rm(input_tensor, gamma_tensor, beta_tensor, epsilon)
    output_torch = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_torch, pcc=0.999)
