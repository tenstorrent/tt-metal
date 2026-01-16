#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, transpose_type, pad_val",
    [
        ([1, 1, 32, 64], "HC", 0),
        # ([1, 1, 32, 64],    "HC", 0),  # duplicate
        ([1, 16, 32, 512], "HC", 0),
        ([32, 1, 16, 64], "CN", 0),
        ([1, 32, 16, 576], "HC", 0),
        ([1, 128, 32, 576], "HC", 0),
        # ([1, 1, 32, 64],    "HC", 0),  # duplicate
        ([1, 32, 1, 64], "HC", 0),
        ([1, 32, 32, 576], "HC", 0),
        ([1, 32, 128, 512], "HC", 0),
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        ([1, 32, 16, 512], "HC", 0),
        ([1, 128, 32, 128], "HC", 0),
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([32, 1, 16, 64],   "CN", None),  # duplicate
        # ([1, 32, 16, 576],  "HC", 0),  # duplicate
        # ([1, 128, 32, 576], "HC", 0),  # duplicate
        # ([1, 1, 32, 64],    "HC", 0),  # duplicate
        # ([1, 32, 1, 64],    "HC", 0),  # duplicate
        # ([1, 32, 32, 576],  "HC", 0),  # duplicate
        # ([1, 32, 128, 512], "HC", 0),  # duplicate
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([1, 32, 16, 512],  "HC", 0),  # duplicate
        # ([1, 128, 32, 128], "HC", 0),  # duplicate
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([32, 1, 16, 64],   "CN", None),  # duplicate
        # ([1, 32, 16, 576],  "HC", 0),  # duplicate
        # ([1, 128, 32, 576], "HC", 0),  # duplicate
        # ([1, 1, 32, 64],    "HC", 0),  # duplicate
        # ([1, 32, 1, 64],    "HC", 0),  # duplicate
        # ([1, 32, 32, 576],  "HC", 0),  # duplicate
        # ([1, 32, 128, 512], "HC", 0),  # duplicate
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([1, 32, 16, 512],  "HC", 0),  # duplicate
        # ([1, 128, 32, 128], "HC", 0),  # duplicate
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([32, 1, 16, 64],   "CN", None),  # duplicate
        # ([1, 32, 16, 576],  "HC", 0),  # duplicate
        # ([1, 128, 32, 576], "HC", 0),  # duplicate
        # ([1, 1, 32, 64],    "HC", 0),  # duplicate
        # ([1, 32, 1, 64],    "HC", 0),  # duplicate
        # ([1, 32, 32, 576],  "HC", 0),  # duplicate
        # ([1, 32, 128, 512], "HC", 0),  # duplicate
        # ([1, 16, 32, 512],  "HC", 0),  # duplicate
        # ([1, 32, 16, 512],  "HC", 0),  # duplicate
        # ([1, 128, 32, 128], "HC", 0),  # duplicate
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_transpose(mesh_device, shape, transpose_type, pad_val, dtype, layout, mem_config, enable_trace):
    if transpose_type == "HC":
        dim2, dim1 = 2, 1
    elif transpose_type == "CN":
        dim2, dim1 = 1, 0
    else:
        assert False, f"Unsupported transpose_type {transpose_type}"

    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.transpose(torch_input, dim1, dim2)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.transpose(tt_input, dim1, dim2, pad_value=pad_val)

    def check_op(tt_output):
        assert_with_pcc(torch_output, tt_output, 0.9999)

    run_test(mesh_device, run_op, check_op, enable_trace)
