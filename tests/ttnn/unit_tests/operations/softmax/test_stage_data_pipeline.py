# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TDD Stage 1: data_pipeline - Reader/writer passthrough with compute identity copy."""

import pytest
import torch
import ttnn

from ttnn.operations.softmax import softmax


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 128),
        (1, 1, 32, 256),
        (4, 2, 64, 64),
    ],
)
def test_stage_data_pipeline(device, shape):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input)
    torch_output = ttnn.to_torch(ttnn_output)

    # Reference: passthrough (identity)
    expected = torch_input.float()
    actual = torch_output.float()

    assert (
        ttnn_output.shape == ttnn_input.shape
    ), f"Shape mismatch: expected {ttnn_input.shape}, got {ttnn_output.shape}"
    assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), f"Data pipeline passthrough failed for shape {shape}"
