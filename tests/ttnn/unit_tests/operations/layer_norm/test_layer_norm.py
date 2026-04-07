# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from .layer_norm import layer_norm


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 32, 96), id="multi_tile"),
    ],
)
def test_layer_norm_shape_scaffold(device, shape):
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, epsilon=1e-5)

    assert list(ttnn_output.shape) == list(shape)
