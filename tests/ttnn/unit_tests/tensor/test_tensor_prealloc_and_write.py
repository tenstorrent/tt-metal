# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED])
@pytest.mark.parametrize("memory_location", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 5), (False, 5)),
)
def test_tensor_preallocation_and_write_apis(
    num_loops, enable_async, in_dtype, mem_layout, memory_location, tensor_layout, device
):
    if in_dtype == ttnn.bfloat8_b and tensor_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Row Major Layout not supported for Bfp8")
    torch.manual_seed(0)
    device.enable_async(enable_async)
    shapes = [(1, 10, 64, 96), (32, 1, 64, 64), (32, 3, 256, 256), (16, 1, 1024, 1024)]

    for tensor_shape in shapes:
        # Preallocate tensor on device
        preallocated_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tensor_shape),
            in_dtype,
            tensor_layout,
            device,
            ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=memory_location),
        )
        for loop in range(num_loops):
            # Write to prreallocated tensor multiple times
            input_tensor_a = torch.randn(tensor_shape).bfloat16()
            tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in_dtype).to(tensor_layout)
            ttnn.copy_host_to_device_tensor(tt_input_tensor_a, preallocated_tensor)
            readback = preallocated_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            allclose, output = comp_pcc(readback, input_tensor_a)
            assert allclose, f"FAILED: {output}"

    device.enable_async(False)
