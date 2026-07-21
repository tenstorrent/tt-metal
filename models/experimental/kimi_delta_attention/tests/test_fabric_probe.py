# SPDX-License-Identifier: Apache-2.0
# Minimal board-health probe: does (2,4) FABRIC_2D open + a TP all_gather run? Isolates HW from KDA code.
import pytest
import torch
import ttnn
from loguru import logger
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores

_F2D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
}


@pytest.mark.parametrize("device_params", [_F2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_probe_all_gather_24(mesh_device):
    logger.info(f"opened mesh {tuple(mesh_device.shape)}")
    g = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    sem = create_global_semaphores(mesh_device, cores, 0)
    bar = ttnn.create_global_semaphore(mesh_device, cores, 0)
    # [1, 8, 32, 64] sharded on tp_axis(1) dim2 -> gather back over tp
    x = ttnn.from_torch(
        torch.randn(1, 8, 32 * 4, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=[None, 2]),
    )
    y = ttnn.experimental.all_gather_async(
        x, dim=2, multi_device_global_semaphore=sem, barrier_semaphore=bar,
        num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear, cluster_axis=1,
    )
    logger.info(f"all_gather OK, out shape {list(y.shape)}")
    assert list(y.shape) == [1, 8, 128, 64]
