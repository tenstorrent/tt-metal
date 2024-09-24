# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


TIMEOUT = 15

# TODO: Missing coverage for mixed precision; passed in dtype does nothing in current matmul path
parameters = {
    "default": {
        "batch_sizes": [(2,)],
        "m_n_sizes": [
            # TODO: Review which cases get triggered for default
            # Single core (won't be hit after padding is added for multicast)
            (32, 32),
            # Multi core (2% math util)
            (320, 384),
            # Multi core reuse (25% math util)
            (512, 512),
            # Multi core reuse multicast in0/in1 (25% math util)
            (4608, 6144),
            # Multi core reuse multicast in0 (25% math util)
            (512, 6144),
            # Multi core reuse multicast in1 (25% math util)
            (4608, 512),
            # Multi core reuse with padding (?% math util)
            (480, 480),
            # Multi core reuse multicast in0/in1 with padding (?% math util)
            (4576, 6112),
            (4416, 6048),
            # Multi core reuse multicast in0 with padding (?% math util)
            (480, 6112),
            (320, 6048),
            # Multi core reuse multicast in1 with padding (?% math util)
            (4576, 480),
            (4416, 320),
        ],
        "k_size": [1024],  # [16, 128, 1024, 4096]
        "batch_matrix_multiply": [True, False],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    }
}


def run(
    batch_sizes,
    m_n_sizes,
    k_size,
    batch_matrix_multiply,
    dtype,
    input_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    (m_size, n_size) = m_n_sizes
    input_a_dtype = dtype
    input_b_dtype = dtype
    input_a_layout = input_layout
    input_b_layout = input_layout

    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.99
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
