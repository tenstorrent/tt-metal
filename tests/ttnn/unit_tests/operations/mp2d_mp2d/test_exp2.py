# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.mp2d_mp2d.utils import DeviceGetter


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("channels", [88])
@pytest.mark.parametrize("input_h", [21])
@pytest.mark.parametrize("input_w", [21])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("padding", [[0, 0]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("ceil_mode", [False])
def test_maxpool2d_mp2d_mp2d(
    batch_size, channels, input_h, input_w, kernel_size, stride, padding, dilation, ceil_mode
):
    device = DeviceGetter.get_device((1, 1))

    # Create torch input tensor: [1, 88, 21, 21]
    torch_input_tensor = torch.ones((batch_size, channels, input_h, input_w), dtype=torch.bfloat16)

    # Permute: [1, 88, 21, 21] -> [1, 21, 21, 88]
    torch_input_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    
    # Reshape: [1, 21, 21, 88] -> [1, 1, 441, 88]
    total_elements = batch_size * input_h * input_w
    torch_input_reshaped = torch.reshape(torch_input_permuted, (1, 1, total_elements, channels))

    # Convert to ttnn tensor
    input_tensor = ttnn.from_torch(torch_input_reshaped, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # First ttnn max_pool2d: [1, 1, 441, 88] -> [1, 1, 100, 88]
    output_tensor_1 = ttnn.max_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )
