# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the pi0.5 prefill all_gather against PR #48301's new ttnn.all_gather.

Four combinations (impl x fabric/mesh), each x {bf16, bf8}:

  test_ag_8dev_async   all_gather_async, FABRIC_1D,      1x8, Topology.Ring  (pi0.5's real AG)
  test_ag_8dev_ttnn    ttnn.all_gather,  FABRIC_1D,      1x8  (-> Linear; new op can't ring here)
  test_ag_ring4_async  all_gather_async, FABRIC_1D_RING, 1x4, Topology.Ring
  test_ag_ring4_ttnn   ttnn.all_gather,  FABRIC_1D_RING, 1x4  (-> Ring)

Gathered output is always [1,1,768,2048]; per-device input is [1,1,768,2048/TP].

Why two meshes: on these chips (TT_VISIBLE_DEVICES=8..15) FABRIC_1D enumerates a 1x8
mesh (8 devices), but FABRIC_1D_RING enumerates only a 2x2 (4 devices) — so the new
op's ring path is only reachable at TP=4. The new ttnn.all_gather derives ring-vs-linear
from the fabric config (FABRIC_1D_RING -> ring, FABRIC_1D -> linear) and ignores the
deprecated op-level topology arg; all_gather_async honors topology=Ring directly.

Run under tracy and read DEVICE KERNEL DURATION (pick a clean DEVICE ID — the marker
bug corrupts some chips with ~4.5e12 ns values):

  TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  python -m tracy -p -r -n ag -o /tmp/tracy_ag $(which pytest) \
    tests/ttnn/unit_tests/operations/ccl/test_ccl_ag_pi05_bench.py -s

Optional: PI0_CCL_PACKET_BYTES=8192 sets the fabric max packet payload (BH max).
See test_ccl_ag_pi05_bench_README.md for measured numbers and interpretation.
"""
import os

import pytest
import torch

import ttnn

_SEQ = 768
_HIDDEN = 2048
_NUM_LINKS = 2
_WARMUP = 6
_ITERS = 24
_MEM = ttnn.L1_MEMORY_CONFIG
_DIM = 3
_DTYPES = [ttnn.bfloat16, ttnn.bfloat8_b]
_DTYPE_IDS = ["bf16", "bf8"]


def _open_mesh(fabric, tp):
    pkt = os.environ.get("PI0_CCL_PACKET_BYTES", "").strip()
    if pkt.isdigit() and int(pkt) > 0:
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(pkt)
        ttnn._ttnn.fabric.set_fabric_config(fabric, router_config=rc)
    else:
        ttnn.set_fabric_config(fabric)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, tp), l1_small_size=24576)


def _input(mesh, tp, dtype):
    return ttnn.from_torch(
        torch.randn(1, 1, _SEQ, _HIDDEN // tp).bfloat16(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=_MEM,
    )


def _warm_and_profile(mesh, run, label):
    for _ in range(_WARMUP):
        run()
    ttnn.synchronize_device(mesh)
    for _ in range(_ITERS):
        run()
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    print(f"\n  {label}: {_ITERS} warm iters — read a clean DEVICE ID's DEVICE KERNEL DURATION from tracy CSV")


def _run_async(mesh, tp, dtype):
    g = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    x = _input(mesh, tp, dtype)
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh, crs, 0)
    out = ttnn.from_torch(
        torch.zeros(1, 1, _SEQ, _HIDDEN), dtype=dtype, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=_MEM)

    def run():
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=out,
            dim=_DIM,
            multi_device_global_semaphore=sems,
            barrier_semaphore=barrier,
            num_links=_NUM_LINKS,
            memory_config=_MEM,
            topology=ttnn.Topology.Ring,
        )

    return run


def _run_ttnn(mesh, tp, dtype):
    x = _input(mesh, tp, dtype)
    return lambda: ttnn.all_gather(x, dim=_DIM, memory_config=_MEM)


def _bench(fabric, tp, kind, dtype, label):
    mesh = _open_mesh(fabric, tp)
    try:
        run = _run_async(mesh, tp, dtype) if kind == "async" else _run_ttnn(mesh, tp, dtype)
        _warm_and_profile(mesh, run, label)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ag_8dev_async(dtype):
    """pi0.5's AG: all_gather_async, FABRIC_1D, 8-device ring (Topology.Ring)."""
    _bench(ttnn.FabricConfig.FABRIC_1D, 8, "async", dtype, "8dev async ring")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ag_8dev_ttnn(dtype):
    """PR #48301 ttnn.all_gather, FABRIC_1D, 8 devices (derives Linear)."""
    _bench(ttnn.FabricConfig.FABRIC_1D, 8, "ttnn", dtype, "8dev ttnn.all_gather (linear)")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ag_ring4_async(dtype):
    """all_gather_async, FABRIC_1D_RING, 4-device ring (Topology.Ring)."""
    _bench(ttnn.FabricConfig.FABRIC_1D_RING, 4, "async", dtype, "ring4 async")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_ag_ring4_ttnn(dtype):
    """PR #48301 ttnn.all_gather, FABRIC_1D_RING, 4-device ring (derives Ring)."""
    _bench(ttnn.FabricConfig.FABRIC_1D_RING, 4, "ttnn", dtype, "ring4 ttnn.all_gather")
