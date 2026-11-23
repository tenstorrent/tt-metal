# Row all-gather (dim 2, column axis, 8-chip ring) throughput by all_gather_async variant: default (4 workers per
# direction per link through a fabric mux), 1 worker per direction (direct connections), broadcast (fabric multicast,
# 1 worker per link: the row-gather transport's scheme).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_ag_variants.py -s     env: AGV_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_ag_variants(mesh_device, device_params):
    mesh = mesh_device
    g = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    k = [0]
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(8, 4))
    for M in [int(m) for m in (os.environ.get("AGV_M") or "4096,16384").split(",")]:
        x = torch.randn(1, 1, M, H)
        tx = ttnn.from_torch(
            x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM, mesh_mapper=rows
        )
        ref = None
        for name, kw in (
            ("default", {}),
            ("wpl1", {"num_workers_per_link": 1}),
            ("wpl2", {"num_workers_per_link": 2}),
            ("broadcast", {"use_broadcast": True}),
        ):

            def ag():
                i = k[0]
                k[0] ^= 1
                return ttnn.experimental.all_gather_async(
                    input_tensor=tx,
                    persistent_output_buffer=None,
                    dim=2,
                    multi_device_global_semaphore=sems[i],
                    num_links=2,
                    barrier_semaphore=bars[i],
                    memory_config=DRAM,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=0,
                    **kw,
                )

            try:
                o = ag()
                ttnn.synchronize_device(mesh)
                t = ttnn.to_torch(ttnn.get_device_tensors(o)[9])
                if ref is None:
                    ref = t
                same = torch.equal(t, ref)
                ttnn.deallocate(o)
                t0 = time.perf_counter()
                for _ in range(10):
                    ttnn.deallocate(ag())
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / 10 * 1e3
                gb = M * H * 1.0625 * 7 / 8 / ms / 1e6
                logger.info(f"AGV M={M} {name}: {ms:.3f} ms, {gb:.1f} GB/s received per chip, matches default {same}")
            except Exception as e:
                logger.warning(f"AGV M={M} {name} ERROR {str(e)[:300]}")
        ttnn.deallocate(tx)
