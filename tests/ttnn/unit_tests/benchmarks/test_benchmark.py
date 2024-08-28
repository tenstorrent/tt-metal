# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
import ttnn
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
# fmt: off
@pytest.mark.parametrize("height,width,average_time", [
    (1024, 1024, 1),
])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
# fmt: on
def test_benchmark_ttnn_add(device, use_program_cache, height, width, dtype, average_time):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((height, width))
    torch_input_tensor_b = torch.rand((height, width))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    ttnn.matmul(input_tensor_a, input_tensor_b)
    total_time = 0
    for i in range(3):
        start = time.time()
        output = ttnn.add(input_tensor_a, input_tensor_b)
        end = time.time()
        duration = end - start
        total_time = total_time + duration
        print(f"ttnn.add: {duration} seconds")
        ttnn.to_torch(output)
    total_time = total_time / 3
    assert total_time <= average_time


@run_for_wormhole_b0()
# fmt: off
@pytest.mark.parametrize("m_size,k_size,n_size,average_time", [
    (384, 1024, 1024, 1),
])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
# fmt: on
def test_benchmark_ttnn_matmul(device, use_program_cache, m_size, k_size, n_size, dtype, average_time):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((m_size, k_size))
    torch_input_tensor_b = torch.rand((k_size, n_size))

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    ttnn.matmul(input_tensor_a, input_tensor_b)
    total_time = 0
    for i in range(3):
        start = time.time()
        output = ttnn.matmul(input_tensor_a, input_tensor_b)
        end = time.time()
        duration = end - start
        total_time = total_time + duration
        print(f"ttnn.matmul: {duration} seconds")
        ttnn.to_torch(output)
    total_time = total_time / 3
    assert total_time <= average_time
