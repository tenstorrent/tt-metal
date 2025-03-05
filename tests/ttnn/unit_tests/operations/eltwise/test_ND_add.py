# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 1, 16, 6, 64, 64], [1, 1, 16, 6, 64, 64]],
        [[1, 16, 8, 49, 49], [1, 16, 1, 49, 49]],
        [[1, 4, 16, 49, 49], [1, 4, 1, 49, 49]],
        [[1, 64, 4, 49, 49], [1, 64, 1, 49, 49]],
        [[1, 2, 8, 6, 32, 64], [1, 2, 8, 6, 32, 64]],
        [[1, 2, 4, 1, 2, 2], [1, 2, 1, 1, 2, 2]],
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 256]],
    ],
)
def test_ND_batch_channel_bcast(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.experimental.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)
    # print(output_tensor)
    # print(torch_output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
