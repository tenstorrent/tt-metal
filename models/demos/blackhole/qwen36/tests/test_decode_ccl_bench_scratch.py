# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: decode-shaped CCL microbenchmark on the 1x4 mesh, trace-timed (per-op device time = replay / K).
AG [1,1,32,1280]->[..,5120] bf16, RS [1,1,32,5120]->[..,1280], composite all-reduce (AG dim0 + sum), row-major AG.
  pytest models/demos/blackhole/qwen36/tests/test_decode_ccl_bench_scratch.py -s
"""
import os
import time

import pytest
import torch

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 256 * 1024 * 1024,
    }
]
K = int(os.environ.get("CCL_BENCH_K", "16"))


def _trace_time(mesh, fn, k=K, reps=5):
    fn()  # compile / warm-up
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(k)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    for o in outs:
        try:
            ttnn.deallocate(o)
        except Exception:
            pass
    return best / k * 1e6


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_decode_ccl_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    max_links = tt_ccl.get_num_links(1)
    print(f"CCL_BENCH max_links={max_links}", flush=True)
    H, TP = 5120, 4
    torch.manual_seed(0)
    x_frac = torch.randn(1, 1, 32, H, dtype=torch.bfloat16)  # will be sharded on dim 3 -> [.., 1280] per device
    ag_in = ttnn.from_torch(
        x_frac,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )
    rs_in = ttnn.from_torch(
        torch.randn(1, 1, 32, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    topos = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}
    mems = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

    def ag(x, links, topo, mem, chunks, workers, bufs=2):
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=links,
            topology=topo,
            memory_config=mem,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=bufs,
        )

    def rs(x, links, topo, mem, chunks, workers, bufs=2):
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=links,
            memory_config=mem,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topo,
            chunks_per_sync=chunks,
            num_workers_per_link=workers,
            num_buffers_per_channel=bufs,
        )

    # correctness of the default AG once
    out = ag(ag_in, 1, ttnn.Topology.Ring, ttnn.DRAM_MEMORY_CONFIG, 10, 2)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    assert torch.allclose(got, x_frac.float(), atol=1e-2), "AG mismatch"
    ttnn.deallocate(out)

    for name, fn, inp in (("AG", ag, ag_in), ("RS", rs, rs_in)):
        for links in sorted({1, max_links}):
            for tname, topo in topos.items():
                for mname, mem in mems.items():
                    for chunks in (1, 4, 10):
                        for workers in (1, 2):
                            try:
                                us = _trace_time(mesh, lambda: fn(inp, links, topo, mem, chunks, workers))
                                print(
                                    f"CCL_BENCH {name} links={links} topo={tname} mem={mname} chunks={chunks} workers={workers} us={us:.1f}",
                                    flush=True,
                                )
                            except Exception as e:
                                print(
                                    f"CCL_BENCH {name} links={links} topo={tname} mem={mname} chunks={chunks} workers={workers} FAILED {str(e).splitlines()[0][:100]}",
                                    flush=True,
                                )
                                ttnn.synchronize_device(mesh)
    # composite all-reduce: gather partials on dim 0 then sum (one CCL)
    for links in sorted({1, max_links}):
        for tname, topo in topos.items():

            def ar():
                g = ttnn.experimental.all_gather_async(
                    rs_in,
                    persistent_output_buffer=None,
                    dim=0,
                    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=links,
                    topology=topo,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
                s = ttnn.sum(g, dim=0, keepdim=True)
                ttnn.deallocate(g)
                return s

            try:
                us = _trace_time(mesh, ar)
                print(f"CCL_BENCH ALLREDUCE(AGdim0+sum) links={links} topo={tname} us={us:.1f}", flush=True)
            except Exception as e:
                print(
                    f"CCL_BENCH ALLREDUCE links={links} topo={tname} FAILED {str(e).splitlines()[0][:100]}", flush=True
                )
    # row-major AG of the single valid row [1,1,1,1280] -> [1,1,1,5120]
    rm_in = ttnn.from_torch(
        x_frac[:, :, :1, :].contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )
    for links in sorted({1, max_links}):
        for tname, topo in topos.items():
            try:
                us = _trace_time(mesh, lambda: ag(rm_in, links, topo, ttnn.DRAM_MEMORY_CONFIG, 1, 1))
                print(f"CCL_BENCH AG_rowmajor_1row links={links} topo={tname} us={us:.1f}", flush=True)
            except Exception as e:
                print(
                    f"CCL_BENCH AG_rowmajor_1row links={links} topo={tname} FAILED {str(e).splitlines()[0][:100]}",
                    flush=True,
                )
    # reference: the small ops around the norm
    xs = ag(ag_in, 1, ttnn.Topology.Ring, ttnn.DRAM_MEMORY_CONFIG, 10, 2)
    w = ttnn.from_torch(
        torch.ones(1, 1, 1, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    us = _trace_time(mesh, lambda: ttnn.rms_norm(xs, weight=w, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG))
    print(f"CCL_BENCH rms_norm_interleaved [1,1,32,5120] us={us:.1f}", flush=True)
    us = _trace_time(
        mesh,
        lambda: ttnn.rms_norm(
            xs, weight=w, epsilon=1e-6, residual_input_tensor=xs, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
    )
    print(f"CCL_BENCH rms_norm_with_residual us={us:.1f}", flush=True)
    us = _trace_time(mesh, lambda: ttnn.add(xs, xs, memory_config=ttnn.L1_MEMORY_CONFIG))
    print(f"CCL_BENCH add [1,1,32,5120] us={us:.1f}", flush=True)
