#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        ([1, 1, 4096, 64], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 4096, 64], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
        ([1, 1, 129280, 224], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ([1, 8, 128, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_untilize(mesh_device, shape, dtype, mem_config, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch_input

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.untilize(tt_input)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "in_shape, out_shape, dtype, mem_config",
    [
        ([1, 1, 32, 4], [1, 1, 32, 4], ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
        ([1, 1, 32, 8], [1, 1, 32, 8], ttnn.uint16, ttnn.L1_MEMORY_CONFIG),
        ([1, 1, 32, 8], [1, 1, 32, 8], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_untilize_with_unpadding(mesh_device, in_shape, out_shape, dtype, mem_config, layout, enable_trace):
    torch_input = random_torch_tensor(dtype, in_shape)
    torch_output = torch_input

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    end_indices = [i - 1 for i in out_shape]

    def run_op():
        return ttnn.untilize_with_unpadding(tt_input, end_indices)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
