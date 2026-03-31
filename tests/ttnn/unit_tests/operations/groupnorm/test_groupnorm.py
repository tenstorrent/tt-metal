# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GroupNorm - Acceptance Tests

Immutable acceptance test for the GroupNorm operation.
Do NOT modify this file. It is the specification.

Run from repo root:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/groupnorm/test_groupnorm.py -v
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm import groupnorm


def pytorch_groupnorm_reference(
    input_tensor: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    PyTorch reference implementation of GroupNorm (without affine parameters).

    Input: (N, 1, H*W, C)
    Groups channels into num_groups groups of group_size = C // num_groups.
    Normalizes each group independently over spatial (H*W) and channel (group_size) dimensions.

    Formula: y = (x - mean) / sqrt(var + eps)
    """
    N, _, HW, C = input_tensor.shape
    group_size = C // num_groups

    # Reshape to (N, num_groups, HW * group_size) for reduction
    x = input_tensor.reshape(N, num_groups, HW * group_size)
    mean = x.mean(dim=2, keepdim=True)
    var = x.var(dim=2, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm.reshape(N, 1, HW, C)


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        # Single tile row, minimal group count
        pytest.param((1, 1, 32, 64), 2, id="single_tile_row_2groups"),
        # Multi tile rows, group_size=32
        pytest.param((1, 1, 64, 128), 4, id="2x4_tiles_4groups"),
        # Larger spatial, larger channel
        pytest.param((1, 1, 128, 256), 8, id="4x8_tiles_8groups"),
        # Non-square tile grid
        pytest.param((1, 1, 32, 256), 8, id="1x8_tiles_8groups"),
        # Multi-batch
        pytest.param((2, 1, 64, 128), 4, id="batch2_4groups"),
        # Large group_size (fewer groups)
        pytest.param((1, 1, 64, 128), 2, id="2groups_groupsize64"),
        # Single group (equivalent to LayerNorm over spatial+channel)
        pytest.param((1, 1, 32, 32), 1, id="single_group_single_tile"),
    ],
)
def test_groupnorm(device, shape, num_groups):
    """Test GroupNorm against PyTorch reference for various shapes and group counts."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = groupnorm(ttnn_input, num_groups=num_groups)

    # Verify shape preserved
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {list(ttnn_output.shape)} vs {list(shape)}"

    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = pytorch_groupnorm_reference(torch_input, num_groups)

    # Multi-step normalization tolerance: rtol=0.05, atol=0.2
    # GroupNorm involves reduce, subtract, square, reduce, rsqrt, multiply — several precision-losing steps
    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=0.05,
        atol=0.2,
    ), (
        f"Output mismatch for shape={shape}, num_groups={num_groups}. "
        f"Max abs diff: {(torch_output.float() - torch_expected.float()).abs().max().item():.6f}, "
        f"Max rel diff: {((torch_output.float() - torch_expected.float()).abs() / (torch_expected.float().abs() + 1e-8)).max().item():.6f}"
    )


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        pytest.param((1, 1, 32, 64), 2, id="constant_input"),
    ],
)
def test_groupnorm_constant_input(device, shape, num_groups):
    """
    Constant input: all elements in each group are equal.
    GroupNorm of constant input should produce all zeros (mean = x, var = 0, x - mean = 0).
    """
    torch.manual_seed(42)
    # Use a constant value per group
    torch_input = torch.ones(shape, dtype=torch.bfloat16) * 3.0

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = groupnorm(ttnn_input, num_groups=num_groups)

    torch_output = ttnn.to_torch(ttnn_output)

    # For constant input, GroupNorm output should be all zeros
    # (x - mean) / sqrt(var + eps) = 0 / sqrt(0 + eps) = 0
    assert torch.allclose(
        torch_output.float(),
        torch.zeros_like(torch_output).float(),
        rtol=0.01,
        atol=0.01,
    ), (
        f"Constant input should produce zero output. " f"Max abs value: {torch_output.float().abs().max().item():.6f}"
    )


@pytest.mark.parametrize(
    "shape, num_groups, eps",
    [
        pytest.param((1, 1, 32, 64), 2, 1e-3, id="eps_1e-3"),
        pytest.param((1, 1, 32, 64), 2, 1e-1, id="eps_1e-1"),
    ],
)
def test_groupnorm_epsilon(device, shape, num_groups, eps):
    """Test GroupNorm with different epsilon values."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = groupnorm(ttnn_input, num_groups=num_groups, epsilon=eps)

    torch_output = ttnn.to_torch(ttnn_output)
    torch_expected = pytorch_groupnorm_reference(torch_input, num_groups, eps=eps)

    assert torch.allclose(
        torch_output.float(),
        torch_expected.float(),
        rtol=0.05,
        atol=0.2,
    ), (
        f"Output mismatch for eps={eps}. "
        f"Max abs diff: {(torch_output.float() - torch_expected.float()).abs().max().item():.6f}"
    )
