# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL device-kernel latency for TP=8 prefill: reduce_scatter + all_gather (Ring topology).

Shapes match pi0.5 prefill MLP (seq=768, hidden=1024, mlp_dim=4096, TP=8):
  hidden   [1,1,768,1024]  — down-proj all_reduce target
  mlp_mid  [1,1,768,4096]  — gate/up scatter target

Wall-clock time is printed by pytest -s.  Device-kernel duration (which
matches the prefill budget) requires running under tracy:

  python -m tracy -p -r -n ccl_prefill_tp8 -o /tmp/tracy_ccl_tp8 \\
    models/experimental/pi0_5/tests/perf/test_ccl_prefill_tp8_perf.py

Requires 8 BH chips + fabric.  Reset with tt-smi -glx_reset if needed.
"""

import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

N_WARMUP = 3
N_ITER = 10
TP = 8

_SHAPES = {
    "hidden": (1, 1, 768, 1024),
    "mlp_mid": (1, 1, 768, 4096),
}

# CCL kwargs matching stage_prefill_tp4.py exactly.
_RS = {
    "num_links": 2,
    "num_workers_per_link": 2,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
    "num_buffers_per_channel": 2,
    "topology": ttnn.Topology.Ring,
}
_AG = {
    "num_links": 2,
    "num_workers_per_link": 4,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
    "num_buffers_per_channel": 2,
    "topology": ttnn.Topology.Ring,
}


def _replicated(t_torch, mesh):
    return ttnn.from_torch(
        t_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _bench_op(name, shape, mesh):
    """Run one op (RS, AG, or RS+AG) traced, report wall-clock µs/call."""
    t_torch = torch.randn(*shape).bfloat16()
    x = _replicated(t_torch, mesh)
    scatter_dim = len(shape) - 1

    # Determine op to benchmark based on name tag
    if name == "reduce_scatter":

        def _op():
            return ttnn.reduce_scatter(x, scatter_dim, **_RS)

    elif name == "all_gather":
        scattered = ttnn.reduce_scatter(x, scatter_dim, **_RS)
        ttnn.synchronize_device(mesh)

        def _op():
            return ttnn.all_gather(scattered, scatter_dim, **_AG)

    else:  # all_reduce

        def _op():
            s = ttnn.reduce_scatter(x, scatter_dim, **_RS)
            out = ttnn.all_gather(s, scatter_dim, **_AG)
            ttnn.deallocate(s)
            return out

    # JIT compile
    out = _op()
    ttnn.synchronize_device(mesh)

    # Trace capture
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = _op()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)

    # Warmup replays
    for _ in range(N_WARMUP):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)

    # Timed replays
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1e6)

    ttnn.ReadDeviceProfiler(mesh)
    ttnn.release_trace(mesh, tid)

    med = statistics.median(times)
    print(f"  {name:16s} shape={list(shape)}  wall-clock median={med:.1f} µs")
    return med


def _run_all(shape_name):
    shape = _SHAPES[shape_name]
    with open_prefill_tp4_mesh(tp=TP, l1_small_size=24576, trace_region_size=134_217_728) as mesh:
        print(f"\n=== CCL TP=8  shape={shape_name} {list(shape)} ===")
        rs_us = _bench_op("reduce_scatter", shape, mesh)
        ag_us = _bench_op("all_gather", shape, mesh)
        ar_us = _bench_op("all_reduce", shape, mesh)
        print(f"  → all_reduce = {ar_us:.1f} µs  (RS {rs_us:.1f} + AG {ag_us:.1f} = {rs_us+ag_us:.1f})")


def test_ccl_prefill_tp8_hidden():
    """RS + AG + all_reduce at hidden shape [1,1,768,1024] on TP=8 Ring."""
    _run_all("hidden")


def test_ccl_prefill_tp8_mlp_mid():
    """RS + AG + all_reduce at MLP-intermediate shape [1,1,768,4096] on TP=8 Ring."""
    _run_all("mlp_mid")
