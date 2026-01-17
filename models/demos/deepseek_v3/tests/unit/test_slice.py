#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, starts, ends, dtype, mem_config",
    [
        ([32, 1, 16, 192], [0, 0, 0, 0], [32, 1, 16, 128], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([32, 1, 16, 192], [0, 0, 0, 128], [32, 1, 16, 192], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 576], [0, 0, 0, 0], [1, 1, 32, 512], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 576], [0, 0, 0, 512], [1, 1, 32, 576], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ###### ([1, 4, 32, 576],   [0, 0, 0, 0],   [1, 4, 32, 576],    ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ([32, 1, 16, 192], [0, 0, 0, 0], [32, 1, 16, 128], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([32, 1, 16, 192], [0, 0, 0, 128], [32, 1, 16, 192], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 0], [1, 1, 32, 512], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 512], [1, 1, 32, 576], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 4, 32, 576],   [0, 0, 0, 0],   [1, 4, 32, 576],    ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([32, 1, 16, 192], [0, 0, 0, 0], [32, 1, 16, 128], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([32, 1, 16, 192], [0, 0, 0, 128], [32, 1, 16, 192], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 0], [1, 1, 32, 512], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 512], [1, 1, 32, 576], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 4, 32, 576],   [0, 0, 0, 0],   [1, 4, 32, 576],    ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([32, 1, 16, 192], [0, 0, 0, 0], [32, 1, 16, 128], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([32, 1, 16, 192], [0, 0, 0, 128], [32, 1, 16, 192], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 0], [1, 1, 32, 512], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 1, 32, 576], [0, 0, 0, 512], [1, 1, 32, 576], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        # ([1, 4, 32, 576],   [0, 0, 0, 0],   [1, 4, 32, 576],    ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        ###### ([1, 32, 8, 32],    [0, 0, 0, 0],   [1, 32, 32, 32],    ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 32, 8, 32],    [0, 0, 0, 0],   [1, 32, 32, 32],    ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 32, 32, 2],    [0, 0, 0, 0],   [1, 32, 32, 32],    ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 32, 32, 2],    [0, 0, 0, 0],   [1, 32, 32, 32],    ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 1, 32, 32],    [0, 0, 0, 0],   [1, 1, 32, 32],     ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 1, 32, 32],    [0, 0, 0, 0],   [1, 1, 32, 32],     ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 1, 32, 32],    [0, 0, 0, 0],   [1, 1, 32, 32],     ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ###### ([1, 1, 32, 32],    [0, 0, 0, 0],   [1, 1, 32, 32],     ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_slice(mesh_device, shape, starts, ends, dtype, mem_config, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2], starts[3] : ends[3]]

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.slice(tt_input, starts, ends)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
