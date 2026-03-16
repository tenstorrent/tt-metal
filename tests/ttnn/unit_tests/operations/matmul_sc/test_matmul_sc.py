# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
matmul_sc Integration Test

Verifies that:
- The operation imports correctly
- The program descriptor creates without errors
- ttnn.generic_op executes without Python-side crashes (stub kernels: output will be garbage)
- Output tensor has the correct shape [M, N]

Run with:
    scripts/tt-test.sh tests/ttnn/unit_tests/operations/matmul_sc/test_matmul_sc.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_sc import matmul_sc


@pytest.mark.parametrize(
    "M, K, N",
    [
        pytest.param(32, 32, 32, id="single_tile"),
        pytest.param(64, 64, 128, id="multi_tile"),
        pytest.param(32, 128, 64, id="non_square"),
    ],
)
def test_matmul_sc_runs(device, M, K, N):
    """Test that matmul_sc executes without errors and produces the correct output shape."""
    torch_a = torch.randn(M, K, dtype=torch.bfloat16)
    torch_b = torch.randn(K, N, dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # With stub kernels, output values will be garbage — that is expected.
    ttnn_output = matmul_sc(ttnn_a, ttnn_b)

    # Shape must be [M, N]
    assert list(ttnn_output.shape) == [M, N], f"Shape mismatch: got {list(ttnn_output.shape)}, expected [{M}, {N}]"
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT


def test_matmul_sc_validation_rank(device):
    """Verify that rank != 2 raises ValueError."""
    torch_a = torch.randn(1, 32, 32, dtype=torch.bfloat16)
    torch_b = torch.randn(32, 32, dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="rank-2"):
        matmul_sc(ttnn_a, ttnn_b)


def test_matmul_sc_validation_inner_dim(device):
    """Verify that mismatched inner dims raises ValueError."""
    torch_a = torch.randn(32, 64, dtype=torch.bfloat16)
    torch_b = torch.randn(32, 32, dtype=torch.bfloat16)  # inner dim mismatch: 64 != 32

    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(ValueError, match="inner dimensions"):
        matmul_sc(ttnn_a, ttnn_b)
