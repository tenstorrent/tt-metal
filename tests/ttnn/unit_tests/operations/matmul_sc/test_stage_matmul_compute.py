# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage: matmul_compute

Full matmul C = A x B using all three helpers:
  read_matmul_tiles, matmul_1d, write_matmul_tiles.

Reference: torch.matmul(A, B)
Tolerances: rtol=0.05, atol=0.2 (bf16 multi-step accumulation)
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_sc import matmul_sc


def pytorch_reference(input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: C = A x B."""
    return torch.matmul(input_a, input_b)


# Shape parametrization: (M, K, N)
# Derived from .tdd_state.json shapes: (1,1,M,K) for A and fixed N per shape.
_SHAPES = {
    "(1, 1, 32, 32)": (32, 32, 32),
    "(1, 1, 64, 128)": (64, 128, 64),
    "(1, 1, 32, 128)": (32, 128, 64),
    "(1, 1, 128, 64)": (128, 64, 128),
    "(1, 1, 32, 256)": (32, 256, 32),
}


@pytest.mark.parametrize(
    "M, K, N",
    [
        pytest.param(32, 32, 32, id="32x32x32"),
        pytest.param(64, 128, 64, id="64x128x64"),
        pytest.param(32, 128, 64, id="32x128x64"),
        pytest.param(128, 64, 128, id="128x64x128"),
        pytest.param(32, 256, 32, id="32x256x32"),
    ],
)
def test_matmul_compute(device, M, K, N):
    """
    Verify full matmul C=AxB matches torch.matmul reference.
    """
    torch.manual_seed(42)

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

    ttnn_output = matmul_sc(ttnn_a, ttnn_b)

    # Output shape check: [M, N]
    assert list(ttnn_output.shape) == [M, N], f"Shape mismatch: {list(ttnn_output.shape)} vs expected [{M}, {N}]"

    # Numerical comparison: output should match torch.matmul(A, B)
    expected = pytorch_reference(torch_a, torch_b)
    torch_output = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        torch_output.float(),
        expected.float(),
        rtol=0.05,
        atol=0.2,
    ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"
