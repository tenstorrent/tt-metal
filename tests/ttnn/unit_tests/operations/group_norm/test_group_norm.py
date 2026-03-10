# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Group Norm - Integration Test

Tests the group_norm operation infrastructure (entry point, program descriptor,
stub kernels). With stub kernels, output values will be garbage but the
operation should execute without Python-side errors and produce correct shapes.

Run from repo root:
    pytest tests/ttnn/unit_tests/operations/group_norm/test_group_norm.py -v
"""

import pytest
import torch
import ttnn

from ttnn.operations.group_norm import group_norm


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="minimal_1group"),
        pytest.param((1, 1, 64, 128), 2, id="2groups_64x128"),
        pytest.param((1, 1, 32, 256), 4, id="4groups_32x256"),
        pytest.param((2, 1, 64, 64), 2, id="batch2_2groups"),
    ],
)
def test_group_norm_runs(device, shape, num_groups):
    """Test that group_norm executes without Python-side errors and output shape is correct."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation (with stub kernels, output will be garbage)
    ttnn_output = group_norm(ttnn_input, num_groups=num_groups)

    # Verify output shape is correct
    expected_shape = list(shape)
    assert list(ttnn_output.shape) == expected_shape, f"Shape mismatch: {list(ttnn_output.shape)} vs {expected_shape}"


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="minimal"),
    ],
)
def test_group_norm_with_gamma_beta(device, shape, num_groups):
    """Test that group_norm accepts explicit gamma and beta tensors."""
    N, _, HW, C = shape
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.ones(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation with explicit gamma/beta
    ttnn_output = group_norm(ttnn_input, num_groups=num_groups, gamma=gamma, beta=beta)

    # Verify output shape is correct
    expected_shape = list(shape)
    assert list(ttnn_output.shape) == expected_shape, f"Shape mismatch: {list(ttnn_output.shape)} vs {expected_shape}"


def test_group_norm_validation_fails(device):
    """Test that validation catches invalid inputs."""
    # C not divisible by num_groups
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="must be divisible by num_groups"):
        group_norm(ttnn_input, num_groups=3)  # 64 not divisible by 3
