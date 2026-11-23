# Row all-gather (dim 2, column axis, 8-chip ring, bf8 [M/8, 8192] per chip) throughput sweep over all_gather_async's
# knobs: workers per link, buffers per channel, chunks per sync. GB/s = bytes received per chip / time.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_ag_tune.py -s     env: AGT_M (16384)
import itertools
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
def test_ag_tune(mesh_device, device_params):
    mesh = mesh_device
    g = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    k = [0]
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(8, 4))
    for M in [int(m) for m in (os.environ.get("AGT_M") or "16384").split(",")]:
        tx = ttnn.from_torch(
            torch.randn(1, 1, M, H),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=DRAM,
            mesh_mapper=rows,
        )
        best = None
        for wpl, bufs, cps in itertools.product((2, 4, 6, 8), (None, 2, 4, 8), (None, 8, 32)):
            kw = {"num_workers_per_link": wpl}
            if bufs:
                kw["num_buffers_per_channel"] = bufs
            if cps:
                kw["chunks_per_sync"] = cps

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
                ttnn.deallocate(ag())
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(8):
                    ttnn.deallocate(ag())
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / 8 * 1e3
                gb = M * H * 1.0625 * 7 / 8 / ms / 1e6
                tag = f"wpl{wpl} bufs{bufs} cps{cps}"
                logger.info(f"AGT M={M} {tag}: {ms:.3f} ms {gb:.1f} GB/s")
                if best is None or ms < best[0]:
                    best = (ms, tag, gb)
            except Exception as e:
                logger.warning(f"AGT M={M} wpl{wpl} bufs{bufs} cps{cps} ERROR {str(e)[:150]}")
        logger.info(f"AGT M={M} BEST {best}")
        ttnn.deallocate(tx)
