# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLP w2 ring-8 reduce: device-kernel duration RS+AG vs all_reduce_async.

Measures ONLY the row-axis (cluster_axis=0) w2 all-reduce at decode sizes on the
real 32-chip BH mesh.  Success = AllReduceAsync device-kernel us/call < RS+AG
with PCC > 0.99 — wall-clock may not move (dispatch-dominated).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_mlp_w2_lar_kernel.py -v -s
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
_N = 1280
_CLUSTER_AXIS = 0
_N_WARMUP = 2
_N_TIMED = 15
_N_INNER = 8

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
    print(f"[{label:<30}] {per_call:7.2f} us/call (traced)")
    return per_call


@pytest.mark.hardware
def test_mlp_w2_reduce_kernel_rsag_vs_async(bh_glx_mesh):
    """w2 post-matmul reduce: RS+AG (DRAM) vs sharded all_reduce_async."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy_v2.tt.llama_ccl import TT_CCL
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    mesh = bh_glx_mesh
    cluster_shape = _MESH_SHAPE
    args = TtQwen36ModelArgs(mesh)
    worker_sub_device_id = ttnn.SubDeviceId(0)
    gs = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gs.x - 1, gs.y - 1))])
    sdm = mesh.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh.load_sub_device_manager(sdm)
    ccl = TT_CCL(mesh, args, worker_sub_device_id=worker_sub_device_id, is_qwen=True, is_qwen36=True)

    torch.manual_seed(1)
    host = torch.randn(*cluster_shape, _M, _N, dtype=torch.bfloat16) * 0.05
    golden = torch.sum(host, dim=_CLUSTER_AXIS)

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

    out = ttnn.all_reduce(x_dram, cluster_axis=_CLUSTER_AXIS, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cluster_shape))
    ok, msg = comp_pcc(golden.float(), got[0].float(), 0.99)
    print(f"  RSAG PCC: {msg}")
    assert ok
    ttnn.deallocate(out)
    rsag_us = _time_traced("RSAG (w2 DRAM path)", run_rsag, mesh)

    assert ccl.qwen36_residual_buffers[0] is not None
    sharded_memcfg = ccl.qwen36_residual_output_memcfgs[0]
    sems = [ttnn.create_global_semaphore(mesh, _SUB_DEVICE_CRS, 0) for _ in range(8)]
    in_cores = 20
    ring = cluster_shape[_CLUSTER_AXIS]
    N_per = _round_up(math.ceil(_N / in_cores), ttnn.TILE_SIZE)
    in_crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), in_cores, _SUB_DEVICE_CRS, row_wise=True
    )
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
        memory_config=sharded_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cluster_shape),
    )
    inter = ttnn.from_torch(
        torch.zeros(*cluster_shape, _M, N_per * in_cores * ring, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=inter_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cluster_shape),
    )
    worker_sub_device = ttnn.SubDevice([_SUB_DEVICE_CRS])
    wsdm = mesh.create_sub_device_manager([worker_sub_device], 1)
    mesh.load_sub_device_manager(wsdm)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    def run_async():
        for _ in range(_N_INNER):
            out = ttnn.experimental.all_reduce_async(
                x_l1,
                inter,
                cluster_axis=_CLUSTER_AXIS,
                mesh_device=mesh,
                multi_device_global_semaphore=sems[0],
                memory_config=sharded_memcfg,
                dtype=ttnn.bfloat16,
                topology=ttnn.Topology.Linear,
                num_links=1,
                subdevice_id=ttnn.SubDeviceId(0),
            )
            ttnn.deallocate(out)

    out = ttnn.experimental.all_reduce_async(
        x_l1,
        inter,
        cluster_axis=_CLUSTER_AXIS,
        mesh_device=mesh,
        multi_device_global_semaphore=sems[0],
        memory_config=sharded_memcfg,
        dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        num_links=1,
        subdevice_id=ttnn.SubDeviceId(0),
    )
    ttnn.synchronize_device(mesh)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cluster_shape))
    ok, msg = comp_pcc(golden.float(), got[0].float(), 0.99)
    print(f"  ASYNC PCC: {msg}")
    assert ok
    ttnn.deallocate(out)
    async_us = _time_traced("ASYNC (w2 LAR path)", run_async, mesh)

    print(f"\n  Device-kernel win: {rsag_us:.1f} -> {async_us:.1f} us/call ({rsag_us/async_us:.2f}x)")
    assert async_us < rsag_us, f"expected async faster: RSAG={rsag_us} ASYNC={async_us}"
    ccl.close()
