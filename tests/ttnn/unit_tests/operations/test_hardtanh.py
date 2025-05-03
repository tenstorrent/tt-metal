# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shapes",
    [[1, 1, 32, 32], [64, 64], [2, 2, 3, 256, 256]],
)
def test_hardtanh_default(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn(shapes[0], dtype=torch.bfloat16) * 10

    golden_fn = ttnn.get_golden_function(ttnn.hardtanh)
    torch_output_tensor = golden_fn(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.hardtanh(input_tensor_a)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.9999


@pytest.mark.parametrize(
    "shapes",
    [[1, 1, 32, 32], [64, 64], [2, 2, 3, 256, 256]],
)
@pytest.mark.parametrize(
    "min",
    [0.25, 0.5, 0.66, -1],
)
@pytest.mark.parametrize(
    "max",
    [1, 2.5, 3, 6.6],
)
def test_hardtanh_args(device, shapes, min, max):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn(shapes[0], dtype=torch.bfloat16) * 10

    golden_fn = ttnn.get_golden_function(ttnn.hardtanh)
    torch_output_tensor = golden_fn(torch_input_tensor_a, min, max)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.hardtanh(input_tensor_a, min, max)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.9999
