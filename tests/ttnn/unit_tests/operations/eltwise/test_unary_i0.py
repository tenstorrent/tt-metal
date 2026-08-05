# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [64, 64],
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-10, 10),
        (10, 85),
        (-85, -10),
    ],
)
def test_i0_range(device, shapes, low, high):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= 0.999
