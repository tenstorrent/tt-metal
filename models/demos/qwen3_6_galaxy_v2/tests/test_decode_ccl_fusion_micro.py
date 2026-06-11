# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode CCL fusion microbenchmark on BH 8x4 GLX — is all_reduce_async faster than RS+AG?

The qwen3.6 decode residual reduces (GDN out_proj, full-attn WO, MLP w2) run on
the row axis (cluster_axis=0, ring=8) and currently use ``ttnn.all_reduce``,
which decomposes on-device into ReduceScatter + AllGather (the profiler shows
both ops). The fused ``ttnn.experimental.all_reduce_async`` exists and is wired
into the model behind flags, but the model's decode path never actually reaches
it (it falls back to RS+AG, and the async residual path was abandoned for
correctness drift over 48 GDN layers).

This microbenchmark answers the prerequisite PERF question in isolation, with
TRACE-REPLAY device timing (eager numbers are dispatch-inflated):

    Is ``all_reduce_async`` materially faster than ``ttnn.all_reduce`` (RS+AG)
    at the exact decode reduce sizes on the real 32-chip BH mesh?

If YES, it justifies investing in the residual-correctness fix to wire it in.
If NO, the RS+AG path is at its practical floor and CCL is not the lever.

Sizes (per-chip, decode):
  - row-axis ring-8 [M=32, N=1280]  (out_proj / WO / MLP-w2 residual reduce)

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_ccl_fusion_micro.py -v -s
"""
from __future__ import annotations

import math
import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_M = 32
_N = 1280  # per-chip residual width (H/4)
_CLUSTER_AXIS = 0  # row ring = 8
_N_WARMUP = 3
_N_TIMED = 20
_N_INNER = 16  # inner calls batched into one trace replay

# Worker sub-device core grid (mirror test_new_all_reduce SUB_DEVICE_CRS, valid on BH 12x10).
_SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=90_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _time_traced(label, run_one, mesh):
    """Capture a trace of _N_INNER calls, replay _N_TIMED times, report us/call."""
    # compile pass (eager) — first run compiles kernels (host write), illegal in trace
    run_one()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    run_one()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(_N_WARMUP):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(_N_TIMED):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    per_call = statistics.mean(times) / _N_INNER
    print(f"[{label:<34}] {statistics.mean(times):>9.1f} us/iter ({per_call:>7.2f} us/call traced, {_N_INNER} calls)")
    return per_call


@pytest.mark.hardware
def test_allreduce_async_vs_rsag_ring8(bh_glx_mesh):
    """Compare ttnn.all_reduce (RS+AG) vs all_reduce_async at ring-8 [32,1280]."""
    mesh = bh_glx_mesh
    cluster_shape = _MESH_SHAPE
    results = {}

    torch.manual_seed(0)
    # host tensor sharded so each of the 8 row-chips holds an independent partial sum
    host = torch.randn(*cluster_shape, _M, _N, dtype=torch.bfloat16) * 0.05
    golden = torch.sum(host, dim=_CLUSTER_AXIS)  # [4, M, N] complete per col

    def _pcc(out_tt, label):
        from models.common.utility_functions import comp_pcc

        got = ttnn.to_torch(
            out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cluster_shape)
        )
        # got: [8,4,M,N] -> take row 0 (all rows replicated after all_reduce)
        got0 = got[0]  # [4, M, N]
        ok, msg = comp_pcc(golden.float(), got0.float(), 0.99)
        print(f"  {label} PCC vs torch sum: {msg}  -> {'PASS' if ok else 'FAIL'}")
        return ok

    # ============ Variant RSAG: ttnn.all_reduce on DRAM (current decode path) ============
    print("\n=== RSAG: ttnn.all_reduce (decomposes to ReduceScatter+AllGather), DRAM, links=1 ===")
    x_dram = ttnn.from_torch(
        host,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cluster_shape),
    )

    def run_rsag():
        for _ in range(_N_INNER):
            out = ttnn.all_reduce(
                x_dram, cluster_axis=_CLUSTER_AXIS, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(out)

    # PCC check (single eager call)
    out = ttnn.all_reduce(x_dram, cluster_axis=_CLUSTER_AXIS, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    results["RSAG_pcc"] = _pcc(out, "RSAG")
    ttnn.deallocate(out)
    try:
        results["RSAG_us"] = _time_traced("RSAG ttnn.all_reduce DRAM l=1", run_rsag, mesh)
    except Exception as e:
        print(f"  RSAG traced timing FAILED: {type(e).__name__}: {str(e)[:160]}")
        results["RSAG_us"] = None

    # ============ Variant ASYNC: ttnn.experimental.all_reduce_async (sharded L1) ============
    print("\n=== ASYNC: ttnn.experimental.all_reduce_async, width-sharded L1, links=1 ===")
    try:
        worker_sub_device = ttnn.SubDevice([_SUB_DEVICE_CRS])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sdm = mesh.create_sub_device_manager([worker_sub_device], 0)
        mesh.load_sub_device_manager(sdm)
        mesh.set_sub_device_stall_group([worker_sub_device_id])

        num_buffers = 8
        sems = [ttnn.create_global_semaphore(mesh, _SUB_DEVICE_CRS, 0) for _ in range(num_buffers)]

        in_cores = 20
        ring = cluster_shape[_CLUSTER_AXIS]
        N_per = _round_up(math.ceil(_N / in_cores), ttnn.TILE_SIZE)
        in_crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), in_cores, _SUB_DEVICE_CRS, row_wise=True
        )
        in_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(in_crs, [_M, N_per], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem = in_mem
        inter_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(in_crs, [_M, N_per * ring], ttnn.ShardOrientation.ROW_MAJOR),
        )
        x_l1 = ttnn.from_torch(
            host,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=in_mem,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cluster_shape),
        )
        inter_host = torch.zeros(*cluster_shape, _M, N_per * in_cores * ring, dtype=torch.bfloat16)
        inters = [
            ttnn.from_torch(
                inter_host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=inter_mem,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cluster_shape),
            )
            for _ in range(num_buffers)
        ]

        _state = {"i": 0}

        def run_async():
            for _ in range(_N_INNER):
                i = _state["i"] % num_buffers
                _state["i"] += 1
                out = ttnn.experimental.all_reduce_async(
                    x_l1,
                    inters[i],
                    cluster_axis=_CLUSTER_AXIS,
                    mesh_device=mesh,
                    multi_device_global_semaphore=sems[i],
                    memory_config=out_mem,
                    dtype=ttnn.bfloat16,
                    topology=ttnn.Topology.Linear,
                    num_links=1,
                    subdevice_id=worker_sub_device_id,
                )
                ttnn.deallocate(out)

        out = ttnn.experimental.all_reduce_async(
            x_l1,
            inters[0],
            cluster_axis=_CLUSTER_AXIS,
            mesh_device=mesh,
            multi_device_global_semaphore=sems[0],
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Linear,
            num_links=1,
            subdevice_id=worker_sub_device_id,
        )
        ttnn.synchronize_device(mesh)
        results["ASYNC_pcc"] = _pcc(out, "ASYNC")
        ttnn.deallocate(out)
        _state["i"] = 0
        results["ASYNC_us"] = _time_traced("ASYNC all_reduce_async L1 l=1", run_async, mesh)
    except Exception as e:
        import traceback

        print(f"  ASYNC FAILED: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
        results["ASYNC_pcc"] = None
        results["ASYNC_us"] = None

    # ============ Summary ============
    print("\n================= RING-8 [32,1280] ALL-REDUCE =================")
    rsag = results.get("RSAG_us")
    asyn = results.get("ASYNC_us")
    print(
        f"  RSAG  (ttnn.all_reduce, current): {rsag if rsag is None else f'{rsag:7.2f} us/call'}  PCC={results.get('RSAG_pcc')}"
    )
    print(
        f"  ASYNC (all_reduce_async, fused) : {asyn if asyn is None else f'{asyn:7.2f} us/call'}  PCC={results.get('ASYNC_pcc')}"
    )
    if rsag and asyn:
        print(f"  speedup (RSAG/ASYNC): {rsag/asyn:.2f}x  ({'ASYNC faster' if asyn<rsag else 'RSAG faster'})")
    print("=" * 62)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
