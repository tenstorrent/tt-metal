# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH microbenchmark: reduce_scatter_minimal_async at the Qwen3.6-27B TP=4 MLP-down shape ([1,1,2048,5120] bf16,
dim=3 -> [.,2048,1280] per device) over num_workers_per_link / chunks_per_sync / num_buffers_per_channel.
  RS_BENCH_ITERS=10 RS_BENCH_S=2048 pytest models/demos/blackhole/qwen36/tests/test_rs_bench_scratch.py -s
Prints one line per config: RS_BENCH workers=.. chunks=.. bufs=.. us=..  (host wall-clock per call, min of iters).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rs_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    iters = int(os.environ.get("RS_BENCH_ITERS", "10"))
    S = int(os.environ.get("RS_BENCH_S", "2048"))
    H = 5120
    x = torch.randn(1, 1, S, H, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    ref = (x.float() * 4).view(1, 1, S, 4, H // 4)  # replicated input: RS sums 4 identical copies
    workers = [int(v) for v in os.environ.get("RS_BENCH_WORKERS", "0,1,2,4").split(",")]
    chunks = [int(v) for v in os.environ.get("RS_BENCH_CHUNKS", "0,1,2,4,8").split(",")]
    bufs = [int(v) for v in os.environ.get("RS_BENCH_BUFS", "2,4,8").split(",")]
    links = int(os.environ.get("RS_BENCH_LINKS", "2"))
    topo = ttnn.Topology.Ring if os.environ.get("RS_BENCH_TOPO", "ring") == "ring" else ttnn.Topology.Linear
    for w in workers:
        for c in chunks:
            for b in bufs:
                kw = dict(
                    num_links=links, topology=topo, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_buffers_per_channel=b
                )
                if w:
                    kw["num_workers_per_link"] = w
                if c:
                    kw["chunks_per_sync"] = c

                def run():
                    return ttnn.experimental.reduce_scatter_minimal_async(
                        tt_x,
                        persistent_output_buffers=None,
                        dim=3,
                        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                        **kw,
                    )

                try:
                    out = run()
                    ttnn.synchronize_device(mesh)
                except Exception as e:  # unsupported combination
                    logger.warning(f"RS_BENCH workers={w} chunks={c} bufs={b} FAILED: {str(e).splitlines()[0][:120]}")
                    continue
                got = (
                    ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
                    .float()
                    .view(1, 1, S, 4, H // 4)
                )
                ok = torch.allclose(got, ref, rtol=2e-2, atol=1e-1)
                ttnn.deallocate(out)
                best = 1e9
                for _ in range(iters):
                    t0 = time.perf_counter()
                    o = run()
                    ttnn.synchronize_device(mesh)
                    best = min(best, time.perf_counter() - t0)
                    ttnn.deallocate(o)
                print(
                    f"RS_BENCH workers={w} chunks={c} bufs={b} links={links} us={best * 1e6:.0f} correct={ok}",
                    flush=True,
                )
