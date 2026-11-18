# Tune ttnn.reduce_scatter (the prefill RS path on BH TP32) at the FF1/FF3 shape on cluster_axis=1 (4-chip ring),
# against the matching all_gather (4096x896 -> 4096x3584, i.e. the same bytes on the wire as the RS).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_rs_tune.py -s --timeout=900
import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE = 10
REPLAYS = 5


def _timed(mesh, fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    first = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(N_PER_TRACE)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, first


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("dtype_name", ["bf8", "bf16"])
def test_rs_tune(mesh_device, device_params, dtype_name):
    dtype = ttnn.bfloat8_b if dtype_name == "bf8" else ttnn.bfloat16
    rows, width = int(os.environ.get("RS_ROWS", "4096")), int(os.environ.get("RS_WIDTH", "3584"))
    tx = torch.randn(1, 1, rows, width, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        tx,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    golden = (tx * 4)[..., : width // 4].float()
    xs = ttnn.from_torch(
        tx[..., : width // 4],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    res = []
    ag_us, _ = _timed(
        mesh_device,
        lambda: ttnn.all_gather(
            xs, dim=3, cluster_axis=1, num_links=2, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring
        ),
    )
    logger.info(f"RSTUNE {dtype_name} all_gather {rows}x{width//4}->{width}: {ag_us:.1f} us")
    workers = [int(v) for v in os.environ.get("RS_WORKERS", "1,2,4,8").split(",")]
    chunks = [None if v == "n" else int(v) for v in os.environ.get("RS_CHUNKS", "n,1,4,16").split(",")]
    bufs = [None if v == "n" else int(v) for v in os.environ.get("RS_BUFS", "n,8").split(",")]
    for w, c, b in itertools.product(workers, chunks, bufs):
        try:
            us, out = _timed(
                mesh_device,
                lambda: ttnn.reduce_scatter(
                    x,
                    3,
                    cluster_axis=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Ring,
                    num_links=2,
                    num_workers_per_link=w,
                    chunks_per_sync=c,
                    num_buffers_per_channel=b,
                ),
            )
            ok, pcc = comp_pcc(golden, out, 0.99)
            res.append((us, w, c, b, pcc))
            logger.info(
                f"RSTUNE {dtype_name} rs w={w} chunks={c} bufs={b}: {us:.1f} us  pcc={pcc:.4f}  ({us/ag_us:.2f}x AG)"
            )
        except Exception as e:
            logger.warning(f"RSTUNE {dtype_name} rs w={w} chunks={c} bufs={b}: ERROR {str(e)[:200]}")
    for us, w, c, b, pcc in sorted(res)[:6]:
        logger.info(f"RSTUNE_BEST {dtype_name} {us:.1f} us w={w} chunks={c} bufs={b} pcc={pcc:.4f}")
