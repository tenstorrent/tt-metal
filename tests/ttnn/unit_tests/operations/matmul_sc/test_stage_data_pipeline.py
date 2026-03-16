# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage: data_pipeline

Reader reads A tiles into cb_in0, compute copies cb_in0 to cb_out (identity),
writer writes cb_out to DRAM. Verifies data pipeline without matmul logic.

Expected output: first M*N elements of A (passthrough of A tiles).
Since K=N for all test shapes here, this is A[:M, :N] = A (all of A).

Shapes used: square (M=K=N), so A shape == B shape == output shape.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_sc import matmul_sc


def pytorch_reference(input_a: torch.Tensor) -> torch.Tensor:
    """
    Stage 1 reference: identity passthrough of A.

    The reader sends Mt*Nt tiles of A sequentially (indices 0..Mt*Nt-1).
    Compute does an identity copy (cb_in0 -> cb_out). Writer writes to C.
    For square shapes where K=N, this is exactly A.
    """
    return input_a


@pytest.mark.parametrize(
    "M, K, N",
    [
        pytest.param(32, 32, 32, id="32x32"),
        pytest.param(64, 64, 64, id="64x64"),
        pytest.param(32, 128, 128, id="32x128"),
        pytest.param(128, 128, 128, id="128x128"),
    ],
)
def test_data_pipeline(device, M, K, N):
    """
    Verify data pipeline stage: A tiles pass through reader->compute->writer unchanged.
    """
    torch.manual_seed(42)

    # A: [M, K], B: [K, N] (square shapes so K=N, B shape is irrelevant for stage 1)
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

    # Numerical comparison: output should match A (identity passthrough)
    expected = pytorch_reference(torch_a)
    torch_output = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        torch_output.float(),
        expected.float(),
        rtol=0.01,
        atol=0.01,
    ), f"Max diff: {(torch_output.float() - expected.float()).abs().max()}"
