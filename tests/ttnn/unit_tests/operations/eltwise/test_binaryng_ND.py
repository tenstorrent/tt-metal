# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import random
import ttnn


@pytest.mark.parametrize(
    "shapes",
    [
        # dims > 4 will be collapsed into a single dim
        [[2, 2, 2, 1, 2, 6, 64, 64], [2, 2, 2, 1, 2, 6, 64, 64]],  # no bcast - rank 8
        [[1, 1, 16, 6, 64, 64], [1, 1, 16, 6, 64, 64]],  # no bcast
        [[1, 2, 8, 6, 32, 64], [1, 2, 8, 6, 32, 64]],
        [[1, 16, 8, 49, 49], [1, 16, 1, 49, 49]],  # channel bcast
        [[1, 4, 16, 49, 49], [1, 4, 1, 49, 49]],
        [[1, 64, 4, 49, 49], [1, 64, 1, 49, 49]],
        [[1, 2, 4, 1, 2, 2], [1, 2, 1, 1, 2, 2]],  # batch bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 256]],
        [[2, 2, 2, 1, 1, 256], [2, 2, 2, 1, 128, 256]],  # row_a bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 2, 1, 1, 256]],  # row_b bcast
        [[2, 2, 2, 1, 128, 1], [2, 2, 1, 1, 128, 256]],  # col_a bcast
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 1]],  # col_b bcast
        [[2, 2, 2, 1, 1, 1], [2, 2, 1, 1, 128, 256]],  # scalar_a
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 1, 1]],  # scalar_b
        [[2, 2, 2, 1, 1, 256], [2, 2, 1, 1, 128, 1]],  # row_a col_b
        [[2, 2, 2, 1, 128, 256], [2, 2, 1, 1, 128, 256]],  # row_b col_A
        [[1, 3, 64, 64, 2], []],
        [[1, 3, 32, 32, 2], []],
        [[1, 3, 16, 16, 2], []],
        [[1, 1, 32, 16, 16], [16]],
        [[1, 1, 32, 16, 16], [16, 16]],
        [[1, 1, 32, 16, 16], [32, 16, 16]],
        [[1, 1, 32, 16, 16], [4, 32, 16, 16]],
        [[1, 1, 1, 1, 2, 6, 64, 64], [64, 64]],
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.divide,
        ttnn.rsub,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
        ttnn.ldexp,
        ttnn.logaddexp,
        ttnn.logaddexp2,
        ttnn.squared_difference,
        ttnn.bias_gelu,
    ],
)
def test_ND_subtile_bcast(device, shapes, ttnn_fn):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16) * 100 - 50
    torch_input_tensor_b = None
    if ttnn_fn == ttnn.divide:
        torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16) * 59 + 1
    else:
        torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16) * 100 - 50

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


@pytest.mark.parametrize(
    "shapes",
    [
        [
            [1, 1, 1, 1, 2, 6, 64, 64],
        ],
        [
            [1, 1, 1, 16, 6, 32, 64],
        ],
        [
            [1, 1, 8, 6, 320, 64],
        ],
        [
            [1, 16, 8, 49, 49],
        ],
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.divide,
        ttnn.rsub,
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
        ttnn.squared_difference,
        ttnn.bias_gelu,
    ],
)
def test_ND_scalar_bcast(device, shapes, ttnn_fn):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16) * 220 - 100
    torch_input_tensor_b = random.randint(-100, 100)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = torch_input_tensor_b

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
