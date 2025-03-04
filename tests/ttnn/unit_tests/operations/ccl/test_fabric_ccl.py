# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import comp_pcc, skip_for_blackhole, run_for_wormhole_b0
import ttnn.experimental


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_sanity(mesh_device):
    pass


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_reduce_scatter(n300_mesh_device):
    torch.manual_seed(2005)
    dim = 3
    input = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
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
    print(sharded_mem_config)
    tt_input = ttnn.from_torch(
        input,
        mesh_mapper=ttnn.ShardTensorToMesh(n300_mesh_device, dim),
        device=n300_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
        dtype=ttnn.bfloat8_b,
    )
    print(tt_input)
    output = ttnn.experimental.llama_reduce_scatter(tt_input, dim, memory_config=sharded_mem_config)
    print(output)
