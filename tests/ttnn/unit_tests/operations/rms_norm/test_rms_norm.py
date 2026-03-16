# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RMS Norm - Integration Test

Run from repo root:
    scripts/tt-test.sh --dev tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def pytorch_rms_norm(input_tensor: torch.Tensor, gamma: torch.Tensor = None, epsilon: float = 1e-6) -> torch.Tensor:
    """PyTorch reference implementation of RMS Norm."""
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + epsilon)
    if gamma is not None:
        normalized = normalized * gamma
    return normalized


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
    ],
)
def test_rms_norm_runs(device, shape, layout):
    """Test rms_norm runs without Python-side errors (stub kernels produce garbage output)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input)

    # Verify output shape is correct
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Verify output layout matches input layout
    assert ttnn_output.layout == layout, f"Layout mismatch: {ttnn_output.layout} vs {layout}"


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="tile"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
    ],
)
def test_rms_norm_with_gamma_runs(device, shape, layout):
    """Test rms_norm with gamma tensor runs without errors (stub kernels produce garbage output)."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma)

    # Verify output shape is correct
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Verify output layout matches input layout
    assert ttnn_output.layout == layout, f"Layout mismatch: {ttnn_output.layout} vs {layout}"


def test_rms_norm_validation_rank(device):
    """Test that rank < 2 raises ValueError."""
    torch_input = torch.randn(32, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        rms_norm(ttnn_input)


def test_rms_norm_validation_gamma_shape(device):
    """Test that mismatched gamma shape raises ValueError."""
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    torch_gamma = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)  # W=32 vs input W=64

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="gamma's last dimension"):
        rms_norm(ttnn_input, gamma=ttnn_gamma)
