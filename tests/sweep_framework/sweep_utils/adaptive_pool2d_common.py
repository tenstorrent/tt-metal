# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from typing import Optional


def run_adaptive_avg_pool2d(
    batch_size: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    output_size: list,
    dtype: ttnn.DataType,
    device: ttnn.Device,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    sharding: Optional[ttnn.TensorMemoryLayout] = None,
):
    """
    Common utility function for running adaptive average pool2d tests

    Args:
        batch_size: Batch dimension size
        input_channels: Number of input channels
        input_height: Input height dimension
        input_width: Input width dimension
        output_size: Target output size [height, width]
        dtype: Data type for computation
        device: Device to run on
        memory_config: Memory configuration
        sharding: Sharding strategy

    Returns:
        PCC value between tt-metal and PyTorch results
    """

    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    torch.manual_seed(0)
    input_shape = [batch_size, input_channels, input_height, input_width]

    # Generate random input tensor
    torch_input_tensor = torch_random(input_shape, low=-100, high=100, dtype=torch.float32)

    if dtype == ttnn.bfloat8_b:
        torch_input_tensor = torch_input_tensor.to(torch.bfloat16)

    # PyTorch reference computation
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, output_size)

    # Convert input to tt-metal format: NCHW -> [1, 1, NHW, C]
    torch_input_tensor_ttnn = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor_ttnn = torch.reshape(
        torch_input_tensor_ttnn, [1, 1, batch_size * input_height * input_width, input_channels]
    )

    # Create tt-metal input tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor_ttnn,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    start_time = start_measuring_time()

    # Call adaptive average pool2d operation
    result = ttnn.adaptive_avg_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
        channels=input_channels,
        output_size=output_size,
        memory_config=memory_config,
    )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # Convert result back to NCHW format for comparison
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    # Check PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    print(f"Adaptive AvgPool2d - Input: {input_shape}, Output size: {output_size}, PCC: {pcc}, Time: {e2e_perf:.2f} us")

    return pcc


def run_adaptive_max_pool2d(
    batch_size: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    output_size: list,
    dtype: ttnn.DataType,
    device: ttnn.Device,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    sharding: Optional[ttnn.TensorMemoryLayout] = None,
):
    """
    Common utility function for running adaptive max pool2d tests

    Args:
        batch_size: Batch dimension size
        input_channels: Number of input channels
        input_height: Input height dimension
        input_width: Input width dimension
        output_size: Target output size [height, width]
        dtype: Data type for computation
        device: Device to run on
        memory_config: Memory configuration
        sharding: Sharding strategy

    Returns:
        PCC value between tt-metal and PyTorch results
    """

    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    torch.manual_seed(0)
    input_shape = [batch_size, input_channels, input_height, input_width]

    # Generate random input tensor
    torch_input_tensor = torch_random(input_shape, low=-100, high=100, dtype=torch.float32)

    if dtype == ttnn.bfloat8_b:
        torch_input_tensor = torch_input_tensor.to(torch.bfloat16)

    # PyTorch reference computation
    torch_output_tensor = torch.nn.functional.adaptive_max_pool2d(torch_input_tensor, output_size)

    # Convert input to tt-metal format: NCHW -> [1, 1, NHW, C]
    torch_input_tensor_ttnn = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor_ttnn = torch.reshape(
        torch_input_tensor_ttnn, [1, 1, batch_size * input_height * input_width, input_channels]
    )

    # Create tt-metal input tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor_ttnn,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    start_time = start_measuring_time()

    # Call adaptive max pool2d operation
    result = ttnn.adaptive_max_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
        channels=input_channels,
        output_size=output_size,
        memory_config=memory_config,
    )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # Convert result back to NCHW format for comparison
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    # Check PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    print(f"Adaptive MaxPool2d - Input: {input_shape}, Output size: {output_size}, PCC: {pcc}, Time: {e2e_perf:.2f} us")

    return pcc


# Device fixture for mesh testing (if needed)
def mesh_device_fixture():
    """Device fixture for multi-device testing"""
    pass  # Implementation would depend on multi-device requirements
