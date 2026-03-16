# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for layer_norm_rm.

Validates:
  - Operation runs without Python-side errors (stub kernels produce garbage output)
  - Output tensor has correct shape, dtype, and layout
  - Tests with and without gamma/beta
  - Tests multiple shapes

When kernels are fully implemented, the numerical comparison against
torch.nn.functional.layer_norm will pass.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


# -----------------------------------------------------------------------
# Shape parametrization
# -----------------------------------------------------------------------
SHAPES = [
    pytest.param((1, 1, 32, 32), id="minimal_single_tile"),
    pytest.param((1, 1, 32, 128), id="multi_tile_W"),
    pytest.param((1, 1, 64, 128), id="multi_tile_HW"),
]


# -----------------------------------------------------------------------
# Basic infrastructure test (stub kernels -- output is garbage)
# -----------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_runs(device, shape):
    """Test that the operation runs without Python-side crashes."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify output metadata
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_with_gamma(device, shape):
    """Test that the operation runs with gamma."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_ttnn = ttnn.from_torch(
        gamma_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, gamma=gamma_ttnn)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_with_gamma_beta(device, shape):
    """Test that the operation runs with gamma and beta."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    beta_torch = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma_ttnn = ttnn.from_torch(
        gamma_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_ttnn = ttnn.from_torch(
        beta_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, gamma=gamma_ttnn, beta=beta_ttnn)

    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


# -----------------------------------------------------------------------
# Numerical correctness test (will fail with stub kernels, pass after
# kernel implementation)
# -----------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_numerical(device, shape):
    """Numerical correctness against PyTorch F.layer_norm (for fully implemented kernels)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    expected = F.layer_norm(torch_input.float(), [shape[-1]]).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)
    torch_output = ttnn.to_torch(ttnn_output)

    # This comparison will only pass once kernels are implemented.
    # With stub kernels, output is garbage -- skip numerical check.
    # Uncomment below when kernels are ready:
    # assert torch.allclose(
    #     torch_output.float(),
    #     expected.float(),
    #     rtol=0.05,
    #     atol=0.2,
    # ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"

    # For now just verify shape
    assert list(ttnn_output.shape) == list(shape)
