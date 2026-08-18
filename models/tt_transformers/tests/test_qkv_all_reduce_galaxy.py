# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce
from tests.ttnn.utils_for_testing import assert_with_pcc

MESH_SHAPE = (8, 4)
GLOBAL_SHAPE = (1, 1, 32, 3072)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=True,
)
def test_llama32_1b_decode_qkv_all_reduce_bh_galaxy(mesh_device):
    torch.manual_seed(0)
    host_input = torch.randn(*GLOBAL_SHAPE, dtype=torch.bfloat16)
    tt_ccl = TT_CCL(mesh_device)
    assert tt_ccl.get_num_links(cluster_axis=1) == 2
    tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1)
    tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1)

    device_input = ttnn.from_torch(
        host_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=MESH_SHAPE),
    )

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(128, 32),
        core_grid=ttnn.CoreGrid(x=6, y=2),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    device_output = tt_all_reduce(
        device_input,
        mesh_device,
        tt_ccl,
        cluster_axis=1,
        dim=0,
        topology=ttnn.Topology.Ring,
        memory_config=output_memory_config,
        sharded=False,
        dtype=ttnn.bfloat8_b,
        use_composite=False,
    )
    ttnn.synchronize_device(mesh_device)

    host_output = ttnn.to_torch(
        device_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=MESH_SHAPE),
    )
    expected = (host_input.float() * MESH_SHAPE[1]).repeat(MESH_SHAPE[1], 1, 1, 1)
    assert_with_pcc(expected, host_output.float(), pcc=0.999)

    device_output.deallocate(True)
    device_input.deallocate(True)
