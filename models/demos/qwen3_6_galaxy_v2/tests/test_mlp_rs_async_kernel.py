# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLP w1/w3 reduce-scatter (cluster_axis=1): SYNC ttnn.reduce_scatter vs async
reduce_scatter_minimal_async, at the qwen3.6 DECODE shape on the real 32-chip mesh.

The decode MLP (_mlp_decode_qwen36) feeds INTERLEAVED DRAM input, which routes
line_reduce_scatter to the SYNC ttnn.reduce_scatter branch (the 26.5us top decode
CCL op). all_reduce/all_gather already use async; reduce_scatter does not. This
isolates whether the async variant is (a) numerically equal (PCC>0.99) and (b)
faster, before wiring it into the decode MLP.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_mlp_rs_async_kernel.py -v -s
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn

_MESH_SHAPE = (8, 4)
_M = 32  # tile-padded decode rows
_N = 2176  # w1 intermediate_per_tp (from layers.*.feed_forward.w1_interleaved [.,.,1280,2176])
_DIM = 3
_CLUSTER_AXIS = 1  # cols (ring of 4)
_RING = _MESH_SHAPE[_CLUSTER_AXIS]
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
    print(f"[{label:<34}] {per_call:7.2f} us/call (traced)")
    return per_call


@pytest.mark.hardware
def test_mlp_rs_sync_vs_async(bh_glx_mesh):
    from models.common.utility_functions import comp_pcc

    mesh = bh_glx_mesh
    cs = _MESH_SHAPE

    torch.manual_seed(1)
    host = torch.randn(*cs, _M, _N, dtype=torch.bfloat16) * 0.05
    # reduce_scatter dim=3 cluster_axis=1: sum over the 4 cols, then each col keeps its 1/RING slice.
    summed = torch.sum(host, dim=_CLUSTER_AXIS)  # [8, M, N]
    Nsc = _N // _RING
    golden = torch.zeros(*cs, _M, Nsc, dtype=torch.bfloat16)
    for c in range(_RING):
        golden[:, c] = summed[:, :, c * Nsc : (c + 1) * Nsc]

    x = ttnn.from_torch(
        host,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=cs),
    )

    def _gather(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=cs))

    # ---- SYNC: ttnn.reduce_scatter (current decode-MLP path), full-grid (no sub-device) ----
    def run_sync():
        for _ in range(_N_INNER):
            o = ttnn.reduce_scatter(
                x,
                _DIM,
                cluster_axis=_CLUSTER_AXIS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                num_links=1,
            )
            ttnn.deallocate(o)

    o = ttnn.reduce_scatter(
        x,
        _DIM,
        cluster_axis=_CLUSTER_AXIS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        num_links=1,
    )
    got = _gather(o)
    ok, msg = comp_pcc(golden.float(), got.float(), 0.99)
    print(f"  SYNC  reduce_scatter PCC: {msg}")
    assert ok, f"sync RS PCC fail: {msg}"
    ttnn.deallocate(o)
    sync_us = _time_traced("SYNC reduce_scatter", run_sync, mesh)

    # ---- ASYNC: reduce_scatter_minimal_async (needs a worker sub-device) ----
    wsdm = mesh.create_sub_device_manager([ttnn.SubDevice([_SUB_DEVICE_CRS])], 1)
    mesh.load_sub_device_manager(wsdm)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    rs_sem = [ttnn.create_global_semaphore(mesh, _SUB_DEVICE_CRS, 0) for _ in range(3)]  # ring RS needs 3
    bar_sem = ttnn.create_global_semaphore(mesh, _SUB_DEVICE_CRS, 0)

    def run_async():
        for _ in range(_N_INNER):
            o = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor=x,
                persistent_output_buffers=None,
                dim=_DIM,
                multi_device_global_semaphore=rs_sem,
                barrier_semaphore=bar_sem,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=ttnn.SubDeviceId(0),
                cluster_axis=_CLUSTER_AXIS,
                num_workers_per_link=1,
            )
            ttnn.deallocate(o)

    o = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor=x,
        persistent_output_buffers=None,
        dim=_DIM,
        multi_device_global_semaphore=rs_sem,
        barrier_semaphore=bar_sem,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        subdevice_id=ttnn.SubDeviceId(0),
        cluster_axis=_CLUSTER_AXIS,
        num_workers_per_link=1,
    )
    got = _gather(o)
    ok, msg = comp_pcc(golden.float(), got.float(), 0.99)
    print(f"  ASYNC reduce_scatter PCC: {msg}")
    assert ok, f"async RS PCC fail: {msg}"
    ttnn.deallocate(o)
    async_us = _time_traced("ASYNC reduce_scatter_minimal", run_async, mesh)

    print(f"\n  RESULT: sync={sync_us:.2f}us  async={async_us:.2f}us  speedup={sync_us/max(async_us,1e-9):.2f}x")
    assert async_us < sync_us, f"async not faster: sync={sync_us} async={async_us}"
