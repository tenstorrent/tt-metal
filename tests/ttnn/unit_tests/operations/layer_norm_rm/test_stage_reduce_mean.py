# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TDD Stage 2: reduce_mean

Compute adds reduce(SUM, REDUCE_ROW) with scaler 1/W to produce row-wise mean.
Output is reduced shape (last dim = 32 tiles-worth padded).

Reference: x.mean(dim=-1, keepdim=True).expand(x.shape[0], x.shape[1], x.shape[2], 32)
Tolerances: rtol=0.02, atol=0.1
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
def test_stage_reduce_mean(device, shape):
    """Stage 2: Row-wise mean reduction."""
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input)

    # Output shape for reduce_mean stage
    output_shape = (shape[0], shape[1], shape[2], 32)

    # Reference: row-wise mean expanded to tile width
    expected = torch_input.float().mean(dim=-1, keepdim=True).expand(shape[0], shape[1], shape[2], 32)
    actual = ttnn.to_torch(ttnn_output)

    assert torch.allclose(
        actual.float(), expected.float(), rtol=0.02, atol=0.1
    ), f"Stage 2 reduce_mean failed. Max diff: {(actual.float() - expected.float()).abs().max()}"
