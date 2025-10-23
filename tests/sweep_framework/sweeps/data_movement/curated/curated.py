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


# Override default timeout (seconds) for hang detection
TIMEOUT = 30


# Define the nightly suite
parameters = {
    "nightly": {
        "batch_sizes": [(4,)],
        "height": [256, 512],
        "width": [256, 512],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "op_name": ['"slice"', "permute", "transpose"],
    },
}


def run(
    batch_sizes,
    height,
    width,
    layout,
    input_dtype,
    input_memory_config,
    output_memory_config,
    op_name,
    *,
    device,
) -> list:
    input_shape = (*batch_sizes, height, width)

    # Generate input tensor
    torch_input = torch_random(input_shape, -1.0, 1.0, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )

    # --- Operation-specific behavior ---
    if op_name == "slice":
        # Slice the tensor roughly in the center
        slice_start_h, slice_end_h = height // 4, 3 * height // 4
        slice_start_w, slice_end_w = width // 4, 3 * width // 4

        torch_output = torch_input[..., slice_start_h:slice_end_h, slice_start_w:slice_end_w]

        start_time = start_measuring_time()
        output_tensor = ttnn.slice(
            ttnn_input,
            [*(0 for _ in batch_sizes), slice_start_h, slice_start_w],
            [*batch_sizes, slice_end_h, slice_end_w],
            memory_config=output_memory_config,
        )
        e2e_perf = stop_measuring_time(start_time)

    elif op_name == "permute":
        # Swap last two dimensions (H <-> W)
        torch_output = torch_input.permute(0, 2, 1)

        start_time = start_measuring_time()
        output_tensor = ttnn.permute(
            ttnn_input,
            (0, 2, 1),
            memory_config=output_memory_config,
        )
        e2e_perf = stop_measuring_time(start_time)

    elif op_name == "transpose":
        # Transpose last two dimensions
        torch_output = torch_input.transpose(-2, -1)

        start_time = start_measuring_time()
        output_tensor = ttnn.transpose(
            ttnn_input,
            -2,
            -1,
            memory_config=output_memory_config,
        )
        e2e_perf = stop_measuring_time(start_time)

    else:
        return [False, f"Unsupported op_name: {op_name}"]

    # Convert to torch and validate
    output_tensor = ttnn.to_torch(output_tensor)
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
