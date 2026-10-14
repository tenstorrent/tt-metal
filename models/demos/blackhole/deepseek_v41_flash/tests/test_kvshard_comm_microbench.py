# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feasibility micro-benchmark for sequence-sharding the (single-head) KV cache over the 8 mesh columns: the communication a decode attention layer would add.
Per device, T = 16 users (32 tile rows): q for its 8 heads [32, 4096] bf16 must be gathered to all 64 heads [32, 32768] (a), every column then attends over its
1/8 of the positions for ALL 64 heads and the partial outputs [32, 32768] (already rescaled to the global max) are summed across the 8 columns, each keeping its
own 8 heads = reduce_scatter (b); the per-head running max / sum [32, 64] fp32 are exchanged first (c, all_gather). Prints KV lines: name = us per call
(20 calls in one trace). Compare with today's 0 us (replicated KV: attention is column-local)."""

import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

REP = 20
AXIS = 1


def traced_us(mesh, fn, n=30):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n / REP
    ttnn.release_trace(mesh, tid)
    return dt * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_kvshard_comm(mesh_device):
    md = mesh_device
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    res = {}
    mk = lambda shape, dt=ttnn.bfloat16: ttnn.from_torch(
        torch.randn(*shape).to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    q = mk((1, 1, 32, 4096))
    part = mk((1, 1, 32, 32768))
    ml = mk((1, 1, 32, 128), ttnn.float32)

    def ag(x, dim=3):
        return ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            cluster_axis=AXIS,
            mesh_device=md,
            topology=ccl.topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    def rs(x):
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl.topology,
            cluster_axis=AXIS,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    for name, fn in (
        ("a_q_all_gather_[32,4096]->[32,32768]", lambda: ag(q)),
        ("b_partial_out_reduce_scatter_[32,32768]->[32,4096]", lambda: rs(part)),
        ("c_max_sum_all_gather_[32,128]fp32", lambda: ag(ml)),
    ):
        try:
            res[name] = traced_us(md, fn)
        except Exception as e:  # noqa: BLE001
            res[name] = f"ERR {str(e)[:200]}"
    if all(isinstance(v, float) for v in res.values()):
        res["TOTAL_added_comm_per_attention_layer"] = sum(res.values())
        res["TOTAL_x40_layers_ms"] = sum(res.values()) * 40 / 1000
    for k, v in res.items():
        print(f"KV {k} = {v if isinstance(v, str) else f'{v:.2f}'}", flush=True)
