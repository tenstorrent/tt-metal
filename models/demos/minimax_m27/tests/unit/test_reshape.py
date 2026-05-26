#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "in_shape, out_shape, layout, mem_config",
    [
        ([1, 1, 32, 3072], [1, 32, 16, 192], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 32, 128, 128], [1, 1, 32, 16384], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 3072], [1, 32, 16, 192], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 32, 128, 128], [1, 1, 32, 16384], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 3072], [1, 32, 16, 192], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 32, 128, 128], [1, 1, 32, 16384], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 3072], [1, 32, 16, 192], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 32, 128, 128], [1, 1, 32, 16384], ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        ([1, 1, 32, 256], [1, 32, 8, 32], ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        ([1, 32, 8, 1], [1, 1, 32, 8], ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
        ([1, 1, 32, 8], [1, 1, 256, 1], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 256, 32], [1, 1, 32, 256], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_reshape(mesh_device, in_shape, out_shape, layout, mem_config, dtype, enable_trace):
    torch_input = random_torch_tensor(dtype, in_shape)
    torch_output = torch_input.reshape(out_shape)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.reshape(tt_input, out_shape)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
