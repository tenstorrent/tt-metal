# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters for adaptive max pool2d testing
parameters = {
    "adaptive_max_pool2d_suite": {
        "input_shape": [
            [1, 64, 224, 224],  # Standard ImageNet input
            [1, 128, 112, 112],  # Mid-layer feature maps
            [1, 256, 56, 56],  # Deeper layer feature maps
            [1, 512, 28, 28],  # Very deep feature maps
            [1, 1024, 14, 14],  # Final feature maps
            [2, 64, 224, 224],  # Batch size 2
            [4, 32, 300, 300],  # Non-standard dimensions
            [1, 96, 150, 150],  # Odd dimensions
            [1, 160, 75, 75],  # Small odd dimensions
            [8, 16, 32, 32],  # Large batch, small spatial
        ],
        "output_size": [
            [1, 1],  # Global adaptive pooling (most common)
            [7, 7],  # Common classifier head size
            [14, 14],  # Intermediate size
            [4, 4],  # Small output
            [2, 2],  # Very small output
            [8, 8],  # Medium output
            [16, 16],  # Larger output
            None,  # Global pooling (should be equivalent to [1, 1])
            [3, 5],  # Asymmetric output size
            [6, 10],  # Another asymmetric case
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_shape = test_vector["input_shape"]
    output_size = test_vector["output_size"]
    input_layout = test_vector["input_a_layout"]
    input_dtype = test_vector["input_a_dtype"]

    # Invalidate bfloat8_b with ROW_MAJOR_LAYOUT
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "bfloat8_b requires TILE_LAYOUT!"

    # Invalidate cases where output size is larger than input size
    if output_size is not None:
        input_h, input_w = input_shape[2], input_shape[3]
        output_h, output_w = output_size[0], output_size[1]

        if output_h > input_h or output_w > input_w:
            return True, f"Adaptive pooling cannot upsample: input {input_h}x{input_w} -> output {output_h}x{output_w}"

    return False, None


def run(
    input_shape,
    output_size,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Adjust input shape for row major layout if needed
    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT and input_shape[-3] % 2 == 1:
        input_shape[-3] += 1

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    # PyTorch adaptive max pooling expects NCHW format
    if output_size is None:
        # Global adaptive pooling - use (1, 1) as output size
        torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor_a, (1, 1))
        effective_output_size = [1, 1]
    else:
        torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor_a, output_size)
        effective_output_size = output_size

    # Convert input tensor to tt-metal format [1, 1, NHW, C]
    [N, C, H, W] = input_shape
    torch_input_tensor_a_ttnn = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_a_ttnn = torch.reshape(torch_input_tensor_a_ttnn, [1, 1, N * H * W, C])

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a_ttnn,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()

    # Call the adaptive max pool operation (this will need to be implemented)
    result = ttnn.adaptive_max_pool2d(
        input_tensor=input_tensor_a,
        batch_size=N,
        input_h=H,
        input_w=W,
        channels=C,
        output_size=effective_output_size,
        memory_config=output_memory_config,
    )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # ttnn operates on channels-last tensors, convert back to NCHW for comparison
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    print(f"PCC: {pcc}, Input shape: {input_shape}, Output size: {effective_output_size}")

    return [pcc, e2e_perf]
