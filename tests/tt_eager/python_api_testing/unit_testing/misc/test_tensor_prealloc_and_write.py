# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn.deprecated as ttl
import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "in_dtype", [ttnn.experimental.tensor.DataType.BFLOAT16, ttnn.experimental.tensor.DataType.BFLOAT8_B]
)
@pytest.mark.parametrize("mem_layout", [ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED])
@pytest.mark.parametrize(
    "memory_location", [ttnn.experimental.tensor.BufferType.L1, ttnn.experimental.tensor.BufferType.DRAM]
)
@pytest.mark.parametrize(
    "tensor_layout", [ttnn.experimental.tensor.Layout.TILE, ttnn.experimental.tensor.Layout.ROW_MAJOR]
)
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 5), (False, 5)),
)
def test_tensor_preallocation_and_write_apis(
    num_loops, enable_async, in_dtype, mem_layout, memory_location, tensor_layout, device
):
    if (
        in_dtype == ttnn.experimental.tensor.DataType.BFLOAT8_B
        and tensor_layout == ttnn.experimental.tensor.Layout.ROW_MAJOR
    ):
        pytest.skip("Row Major Layout not supported for Bfp8")
    torch.manual_seed(0)
    device.enable_async(enable_async)
    shapes = [(1, 10, 64, 96), (32, 1, 64, 64), (32, 3, 256, 256), (16, 1, 1024, 1024)]

    for tensor_shape in shapes:
        # Preallocate tensor on device
        preallocated_tensor = ttnn.experimental.tensor.allocate_tensor_on_device(
            ttnn.Shape(tensor_shape),
            in_dtype,
            tensor_layout,
            device,
            ttnn.experimental.tensor.MemoryConfig(memory_layout=mem_layout, buffer_type=memory_location),
        )
        for loop in range(num_loops):
            # Write to prreallocated tensor multiple times
            input_tensor_a = torch.randn(tensor_shape).bfloat16()
            tt_input_tensor_a = ttnn.experimental.tensor.Tensor(input_tensor_a, in_dtype).to(tensor_layout)
            ttnn.experimental.tensor.write_tensor(tt_input_tensor_a, preallocated_tensor)
            readback = preallocated_tensor.cpu().to(ttnn.experimental.tensor.Layout.ROW_MAJOR).to_torch()
            allclose, output = comp_pcc(readback, input_tensor_a)
            assert allclose, f"FAILED: {output}"

    device.enable_async(False)
