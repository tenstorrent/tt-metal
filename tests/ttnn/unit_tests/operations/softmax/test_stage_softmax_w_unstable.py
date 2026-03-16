# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Auto-generated TDD stage test: softmax_w_unstable
# Softmax dim=-1, numeric_stable=False

import pytest
import torch
import ttnn

from .softmax import softmax


def pytorch_reference(input_tensor):
    return torch.softmax(input_tensor, dim=-1)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square"),
        pytest.param((4, 2, 64, 64), id="multi_batch"),
    ],
)
def test_softmax_w_unstable(device, shape):
    layout = ttnn.TILE_LAYOUT
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    expected = pytorch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=False)

    expected_shape = list(shape)
    assert list(ttnn_output.shape) == expected_shape

    torch_output = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output.float(), expected.float(), rtol=0.05, atol=0.2)
