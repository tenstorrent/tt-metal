# Microbenchmark: the decode MLP reduce-scatter of Llama-70B TP32 on the BH galaxy (reduce_scatter_minimal_async,
# [1,1,ROWS,3584] bf8 DRAM-interleaved, cluster_axis=1 over the 4 devices of a mesh row), swept over
# topology / links / workers per link / chunks per sync / buffers per channel, plus the 64-row (w1+w3 merged) case.
# Each config: 40 calls captured in one trace, replayed 5x; reports device time per call. Also checks the result.
import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE = 40
REPLAYS = 5


def _run(mesh, x, cfg, sems, barrier):
    return ttnn.experimental.reduce_scatter_minimal_async(
        x,
        dim=3,
        multi_device_global_semaphore=sems,
        barrier_semaphore=barrier,
        num_links=cfg["links"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=cfg["topology"],
        cluster_axis=1,
        chunks_per_sync=cfg["chunks"],
        num_workers_per_link=cfg["workers"],
        num_buffers_per_channel=cfg["buffers"],
    )


def _bench(mesh, rows, cfg, sem_sets, barriers):
    torch_x = torch.randn(1, 1, rows, 3584 * 4 // 4 * 1, dtype=torch.bfloat16)  # per-device [rows, 3584]
    x = ttnn.from_torch(
        torch_x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # compile + correctness
    out = _run(mesh, x, cfg, sem_sets[0], barriers[0])
    ttnn.synchronize_device(mesh)
    dev0 = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ref = (4 * ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float())[..., : dev0.shape[-1]]
    ok, pcc = comp_pcc(ref, dev0, 0.99)
    ttnn.deallocate(out)
    # trace N calls
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    for i in range(N_PER_TRACE):
        outs.append(_run(mesh, x, cfg, sem_sets[i % len(sem_sets)], barriers[i % len(barriers)]))
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    ttnn.deallocate(x)
    return us, ok, pcc


FABRICS = {
    "1D": ttnn.FabricConfig.FABRIC_1D,
    "1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
}


@pytest.mark.parametrize(
    "device_params, fabric_name",
    [
        ({"fabric_config": FABRICS["1D"], "trace_region_size": 50_000_000}, "1D"),
        ({"fabric_config": FABRICS["1D_RING"], "trace_region_size": 50_000_000}, "1D_RING"),
    ],
    indirect=["device_params"],
    ids=["fabric1d", "fabric1dring"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_rs_sweep(mesh_device, device_params, fabric_name):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sem_sets = [[ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)] for _ in range(4)]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(4)]
    topologies = [ttnn.Topology.Linear] + ([ttnn.Topology.Ring] if fabric_name == "1D_RING" else [])
    grid_cfg = list(itertools.product(topologies, [1, 2], [None, 1, 2, 4], [None, 2, 8], [None, 4], [32, 64]))
    if os.environ.get("RS_QUICK") == "1":
        grid_cfg = [c for c in grid_cfg if c[4] is None]
    for topo, links, workers, chunks, buffers, rows in grid_cfg:
        cfg = dict(topology=topo, links=links, workers=workers, chunks=chunks, buffers=buffers)
        try:
            us, ok, pcc = _bench(mesh_device, rows, cfg, sem_sets, barriers)
            res = f"{us:7.2f} us/call  correct={ok}"
        except Exception as e:
            res = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        logger.info(
            f"RSWEEP fabric={fabric_name} topo={str(topo).split('.')[-1]} links={links} workers={workers} "
            f"chunks={chunks} buffers={buffers} rows={rows}: {res}"
        )
