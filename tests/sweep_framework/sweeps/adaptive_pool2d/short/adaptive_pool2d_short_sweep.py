# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch
import math

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Short sweep for adaptive pooling operations - focused on commonly used configurations
parameters = {
    "adaptive_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "pool_type": ["avg", "max"],
        "input_specs": [
            # Contains following parameters:
            # [batch_size, input_channels, input_height, input_width, output_height, output_width]
            # Common scenarios from real models
            # [1, 64, 224, 224, 1, 1],  # Global pooling on ImageNet input OOM
            # [1, 128, 112, 112, 1, 1],  # Global pooling mid-network OOM
            [1, 256, 56, 56, 1, 1],  # Global pooling deeper network
            [1, 512, 28, 28, 1, 1],  # Global pooling very deep
            [1, 1024, 14, 14, 1, 1],  # Global pooling final layers
            [1, 512, 7, 7, 7, 7],  # No-op case (input == output size)
            [1, 2048, 7, 7, 1, 1],  # ResNet-style global pooling
            [1, 64, 224, 224, 7, 7],  # Standard classifier head
            [1, 256, 32, 32, 4, 4],  # 8x downsampling
            [1, 128, 64, 64, 8, 8],  # 8x downsampling different size
            # [2, 64, 112, 112, 1, 1],  # Batch size 2 OOM
            # [4, 32, 64, 64, 2, 2],  # Batch size 4 OOM
            # [1, 96, 150, 150, 5, 5],  # Odd dimensions OOM
            [1, 160, 75, 75, 3, 3],  # Small odd dimensions
            [1, 320, 28, 28, 14, 14],  # 2x downsampling
            [1, 480, 14, 14, 7, 7],  # 2x downsampling small
            # Asymmetric output sizes (real-world scenarios)
            [1, 256, 64, 64, 3, 5],  # Asymmetric pooling
            [1, 512, 32, 32, 2, 4],  # Another asymmetric case
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run_adaptive_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, pool_type, dtype, device):
    """Helper function to run adaptive pooling operations"""

    # Generate input tensor
    torch.manual_seed(0)
    input_shape = [in_n, in_c, in_h, in_w]
    torch_input_tensor = torch_random(input_shape, low=-100, high=100, dtype=torch.float32)

    if dtype == ttnn.bfloat8_b:
        torch_input_tensor = torch_input_tensor.to(torch.bfloat16)

    # PyTorch reference
    output_size = (out_h, out_w)
    if pool_type == "avg":
        torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, output_size)
    else:  # max
        torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor, output_size)

    # Convert to tt-metal format [1, 1, NHW, C]
    torch_input_tensor_ttnn = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor_ttnn = torch.reshape(torch_input_tensor_ttnn, [1, 1, in_n * in_h * in_w, in_c])

    # Create tt-metal tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor_ttnn,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start_time = start_measuring_time()

    # Call adaptive pooling operation
    if pool_type == "avg":
        result = ttnn.adaptive_avg_pool2d(
            input_tensor=input_tensor,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:  # max
        result = ttnn.adaptive_max_pool2d(
            input_tensor=input_tensor,
            batch_size=in_n,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            output_size=[out_h, out_w],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # Convert back to NCHW format for comparison
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    # Check results
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    print(f"Adaptive {pool_type} pool2d - Input: {input_shape}, Output size: {output_size}, PCC: {pcc}")

    return pcc


def run(
    input_specs,
    pool_type,
    dtype,
    *,
    device,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
    ) = input_specs

    return run_adaptive_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, pool_type, dtype, device)


import pytest


@pytest.mark.parametrize("input_spec", parameters["adaptive_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("pool_type", parameters["adaptive_pool2d_short_sweep_suite"]["pool_type"])
@pytest.mark.parametrize("dtype", parameters["adaptive_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_adaptive_pool2d_localrun(device, dtype, pool_type, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
    ) = input_spec

    run_adaptive_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        pool_type,
        dtype,
        device,
    )
