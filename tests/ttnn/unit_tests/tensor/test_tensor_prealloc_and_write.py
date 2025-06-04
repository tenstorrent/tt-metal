# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize("shape", [(1, 10, 64, 96), (32, 1, 64, 64), (32, 3, 256, 256), (16, 1, 1024, 1024)])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED])
@pytest.mark.parametrize("memory_location", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_tensor_preallocation_and_write_apis(shape, in_dtype, mem_layout, memory_location, tensor_layout, device):
    if in_dtype == ttnn.bfloat8_b and tensor_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Row Major Layout not supported for Bfp8")
    torch.manual_seed(0)

    # Preallocate tensor on device
    preallocated_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        in_dtype,
        tensor_layout,
        device,
        ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=memory_location),
    )
    for loop in range(5):
        # Write to prreallocated tensor multiple times
        input_tensor_a = torch.randn(shape).bfloat16()
        tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in_dtype).to(tensor_layout)
        ttnn.copy_host_to_device_tensor(tt_input_tensor_a, preallocated_tensor)
        readback = preallocated_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        allclose, output = comp_pcc(readback, input_tensor_a)
        assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize("shape", [(1, 10, 64, 96), (32, 3, 256, 256)])
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED])
@pytest.mark.parametrize("memory_location", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("mesh_device", ((1, 1), 4), indirect=True)
def test_tensor_preallocation_and_write_apis(shape, in_dtype, mem_layout, memory_location, tensor_layout, mesh_device):
    if in_dtype == ttnn.bfloat8_b and tensor_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Row Major Layout not supported for Bfp8")
    torch.manual_seed(0)
    shapes = [(1, 10, 64, 96), (32, 1, 64, 64), (32, 3, 256, 256), (16, 1, 1024, 1024)]

    # Preallocate tensor on device
    preallocated_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        in_dtype,
        tensor_layout,
        mesh_device,
        ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=memory_location),
    )
    for loop in range(5):
        # Write to prreallocated tensor multiple times
        input_tensor_a = torch.randn(shape).bfloat16()
        tt_input_tensor_a = ttnn.from_torch(
            input_tensor_a,
            dtype=in_dtype,
            layout=tensor_layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.copy_host_to_device_tensor(tt_input_tensor_a, preallocated_tensor)
        readback_tensors = [
            ttnn.to_torch(shard)
            for shard in ttnn.get_device_tensors(preallocated_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
        ]
        for readback_tensor in readback_tensors:
            allclose, output = comp_pcc(readback_tensor, input_tensor_a)
            assert allclose, f"FAILED: {output}"
