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
        ([1, 1, 32, 256], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 32, 256], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 8, 128, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        ([8, 1, 32, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        # ([8, 1, 32, 7168], ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),  # duplicate
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_tilize(mesh_device, shape, dtype, mem_config, layout, enable_trace):
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
        return ttnn.tilize(tt_input)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "in_shape, out_shape, pad_val, dtype, mem_config",
    [
        ([1, 1, 32, 8], [1, 1, 32, 32], 0, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        # ([1, 1, 32, 8], [1, 1, 32, 32], 0, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),  # duplicate
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_tilize_with_val_padding(mesh_device, in_shape, out_shape, pad_val, dtype, mem_config, layout, enable_trace):
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

    def run_op():
        return ttnn.tilize_with_val_padding(tt_input, out_shape, pad_val)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
