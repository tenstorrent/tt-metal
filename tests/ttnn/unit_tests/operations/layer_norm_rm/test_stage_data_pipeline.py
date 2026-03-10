# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 1: data_pipeline

Reader reads RM input sticks, compute tilizes then immediately untilizes,
writer writes RM output sticks. Identity passthrough verifies the data
pipeline end-to-end.

Reference: x (identity passthrough)
Tolerances: rtol=0.01, atol=0.01
"""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_rm import layer_norm_rm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="minimal"),
        pytest.param((1, 1, 64, 128), id="multi_tile_hw"),
        pytest.param((1, 1, 32, 256), id="wide"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_stage_data_pipeline(device, shape):
    """Stage 1: Identity passthrough via tilize + untilize."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Verify shape
    assert list(ttnn_output.shape) == list(shape), f"Shape mismatch: {ttnn_output.shape} vs {shape}"

    # Reference: identity passthrough
    expected = torch_input
    actual = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        actual.float(), expected.float(), rtol=0.01, atol=0.01
    ), f"Stage 1 data_pipeline failed. Max diff: {(actual.float() - expected.float()).abs().max()}"
