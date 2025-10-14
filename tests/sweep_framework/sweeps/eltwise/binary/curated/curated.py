# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    start_measuring_time,
    stop_measuring_time,
)
from models.common.utility_functions import torch_random


# Override default timeout (in seconds) for hang detection
TIMEOUT = 30


# Parameter suite
parameters = {
    "nightly": {
        "batch_sizes": [(4,)],
        "height": [384],
        "width": [4096],
        "broadcast": [None, "h", "w", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "op_name": ["add", "sub", "mul", "div"],
    }
}


# Main run function
def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    op_name,
    *,
    device,
) -> list:
    # Define shapes
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)

    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    # Define value ranges for numerical stability
    if op_name == "div":
        low, high = 0.1, 10.0
    else:
        low, high = -3.0, 3.0

    # Create random inputs
    torch_input_a = torch_random(input_shape_a, low, high, dtype=torch.float32)
    torch_input_b = torch_random(input_shape_b, low, high, dtype=torch.float32)

    # Get the TTNN op and its golden function
    ttnn_op = getattr(ttnn, op_name)
    torch_op = ttnn.get_golden_function(ttnn_op)

    # Golden (PyTorch) reference
    torch_output = torch_op(torch_input_a, torch_input_b)

    # Convert inputs to TTNN tensors
    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    # Run op and measure performance
    start_time = start_measuring_time()
    output_tensor = ttnn_op(input_a, input_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Return results and timing
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
