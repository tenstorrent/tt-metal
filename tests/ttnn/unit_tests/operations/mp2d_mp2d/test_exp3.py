# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.mp2d_mp2d.utils import DeviceGetter


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("channels", [88])
@pytest.mark.parametrize("input_h", [10])
@pytest.mark.parametrize("input_w", [10])
@pytest.mark.parametrize("kernel_size", [[3, 3]])
@pytest.mark.parametrize("stride", [[2, 2]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("ceil_mode", [False])
def test_maxpool2d_mp2d_mp2d(
    batch_size, channels, input_h, input_w, kernel_size, stride,  dilation, ceil_mode
):
    device = DeviceGetter.get_device((1, 1))

    # Create torch input tensor: [1, 1, 100, 88]
    torch_input_tensor = torch.ones((batch_size, 1, input_h * input_w, channels), dtype=torch.bfloat16)
    
    # Convert to ttnn tensor
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.Layout.ROW_MAJOR, device=device)

    # Second ttnn max_pool2d: [1, 1, 100, 88] -> [1, 1, 25, 88]
    output_h = 10
    output_w = 10
    padding_2nd = [0, 1, 0, 1]
    
    output_tensor_2 = ttnn.max_pool2d(
        input_tensor=input_tensor,
        batch_size=batch_size,
        input_h=output_h,
        input_w=output_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding_2nd,
        dilation=dilation,
        ceil_mode=ceil_mode,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )

