# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import comp_pcc, skip_for_blackhole, run_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    create_global_semaphore_with_same_address,
)


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_sanity(mesh_device):
    pass


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_reduce_scatter(n300_mesh_device):
    torch.manual_seed(2005)
    dim = 3
    input = torch.ones((1, 1, 32, 128), dtype=torch.bfloat16)
    input[:, :, :, :64] = 2
    compute_grid_size = n300_mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    # create global semaphore handles
    semaphores = create_global_semaphore_with_same_address(n300_mesh_device, ccl_sub_device_crs, 0)

    print(input)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        (32, 32),
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(
        input,
        mesh_mapper=ttnn.ShardTensorToMesh(n300_mesh_device, dim),
        device=n300_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
        dtype=ttnn.bfloat8_b,
    )
    # print(tt_input)
    output = ttnn.experimental.llama_reduce_scatter(tt_input, dim, semaphores)
    # print(tt_input)
    jank_input = ttnn.to_torch(tt_input, mesh_composer=ttnn.ConcatMeshToTensor(n300_mesh_device, dim=dim))
