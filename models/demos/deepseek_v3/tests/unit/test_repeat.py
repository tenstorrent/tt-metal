#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def random_torch_tensor(dtype, shape):
    torch.manual_seed(1234)
    if dtype == ttnn.int32 or dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, repeat_shape, dtype, mem_config",
    [
        ([1, 1, 1, 256], [1, 32, 1, 1], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 4], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 256, 1], [1, 1, 1, 32], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 1, 8], [1, 32, 1, 1], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 128, 7168], [1, 8, 1, 1], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ([1, 1, 32, 8], [1, 7168, 1, 1], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_repeat(mesh_device, shape, repeat_shape, dtype, mem_config, layout):
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input.repeat(repeat_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = ttnn.repeat(tt_input, repeat_shape)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Every device executes the same op, check that each device returned the
    # same result
    for dev in range(mesh_device.get_num_devices()):
        i = dev * shape[0]
        j = (dev + 1) * shape[0]
        assert_with_pcc(torch_output, tt_output[i:j, ...], 0.9999)
