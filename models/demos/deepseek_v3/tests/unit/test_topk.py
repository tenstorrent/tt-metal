#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import random_torch_tensor, run_test
from tests.ttnn.utils_for_testing import assert_with_pcc

K_VALUE = 32
TOPK_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 9))])


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        [32, 8, 64],
        [1, 32, 64],
        [1, 32, 256],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 10000, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_topk(mesh_device, shape, dtype, enable_trace, device_params):
    torch_input = random_torch_tensor(dtype, shape)
    torch_values, _ = torch.topk(torch_input, k=K_VALUE, dim=-1, largest=True, sorted=True)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=TOPK_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        tt_values, _ = ttnn.topk(
            tt_input,
            k=K_VALUE,
            dim=-1,
            largest=True,
            sorted=True,
            memory_config=TOPK_MEMORY_CONFIG,
            sub_core_grids=SUB_CORE_GRIDS,
        )
        return tt_values

    def check_op(tt_output):
        assert_with_pcc(torch_values, tt_output, 0.999)

    run_test(mesh_device, run_op, check_op, enable_trace)
