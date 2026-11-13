# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_reduce_async(one_shot=True): reduce-scatter and all-gather in one op.

The shape is the Qwen3.8-27B TP=8 decode residual: [1, 1, rows, 5120] bf16, L1 width-sharded over a 10x4 core grid
(4 tiles per core), with a persistent [32, 5120 * 8] buffer on the same grid. The default mode multicasts every
device's whole partial to every peer. one_shot sends each owner only its slice; the owner reduces it with the same
kernel and slot order and multicasts the reduced slice back, so the result must equal the default mode's bytes.

Run on 8 chips:
  pytest tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_one_shot.py
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

GRID = (10, 4)
DIM = 5120


def _fabric_params(fabric_config, trace_bytes=8 << 20):
    return {"fabric_config": fabric_config, "trace_region_size": trace_bytes}


FABRICS = [
    pytest.param(_fabric_params(ttnn.FabricConfig.FABRIC_1D), id="fabric_1d"),
    pytest.param(_fabric_params(ttnn.FabricConfig.FABRIC_1D_RING), id="fabric_1d_ring"),
]
MESH_1X8 = [pytest.param((1, 8), id="1x8")]
TOPOLOGIES = [pytest.param(ttnn.Topology.Ring, id="ring"), pytest.param(ttnn.Topology.Linear, id="linear")]


def _width_mem(shard_w, grid_xy=GRID):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_xy[0] - 1, grid_xy[1] - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _setup(mesh_device, grid_xy=GRID):
    tp = list(mesh_device.shape)[1]
    ncores = grid_xy[0] * grid_xy[1]
    mem = _width_mem(DIM // ncores, grid_xy)
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(2)]
    buf = ttnn.from_torch(
        torch.zeros([1, 1, ttnn.TILE_SIZE, DIM * tp]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=_width_mem(DIM * tp // ncores, grid_xy),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return tp, mem, sems, buf


def _partials_host(mesh_device, full):
    """full [1, tp, rows, DIM]: device d gets full[:, d] as [1, 1, rows, DIM]."""
    return ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1)
    )


def _replicated_host(mesh_device, t):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )


def _random(tp, rows, seed):
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(1, tp, rows, DIM, generator=g).to(torch.bfloat16)
    res = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)
    return full, res


def _ar(mesh_device, inp, buf, sem, mem, one_shot, residual=None, topology=ttnn.Topology.Ring):
    kwargs = {"one_shot": True} if one_shot else {}
    if residual is not None:
        kwargs["residual_tensor"] = residual
    return ttnn.experimental.all_reduce_async(
        inp,
        buf,
        cluster_axis=1,
        mesh_device=mesh_device,
        multi_device_global_semaphore=sem,
        topology=topology,
        num_links=2,
        memory_config=mem,
        **kwargs,
    )


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _assert_same(got, ref, what):
    for dev, (g, r) in enumerate(zip(got, ref)):
        assert g.shape == r.shape, f"{what}, device {dev}: shape {tuple(g.shape)} vs {tuple(r.shape)}"
        assert torch.equal(g, r), (
            f"{what}, device {dev}: max |diff| {(g.float() - r.float()).abs().max().item():.3e}, "
            f"{(g != r).sum().item()} elements differ"
        )


@pytest.mark.parametrize("device_params", FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "residual"])
@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32])
def test_all_reduce_async_one_shot_matches_default(mesh_device, topology, fused, rows):
    """one_shot == default mode byte for byte on every device, eager and on program-cache hits."""
    tp, mem, sems, buf = _setup(mesh_device)
    full, res = _random(tp, rows, seed=rows + 100 * fused)
    inp = _partials_host(mesh_device, full).to(mesh_device, mem)
    residual = _replicated_host(mesh_device, res).to(mesh_device, mem) if fused else None

    outs = {}
    for i, (name, one_shot) in enumerate((("default", False), ("one_shot", True), ("one_shot#2", True))):
        out = _ar(mesh_device, inp, buf, sems[i % 2], mem, one_shot, residual, topology)
        outs[name] = _per_device(out)
        ttnn.deallocate(out)
    _assert_same(outs["one_shot"], outs["default"], f"rows={rows} fused={fused}: one_shot vs default")
    _assert_same(outs["one_shot#2"], outs["one_shot"], f"rows={rows} fused={fused}: cached one_shot call")
    for dev in range(1, tp):
        assert torch.equal(outs["one_shot"][dev], outs["one_shot"][0]), f"device {dev} differs from device 0"
    expected = full.float().sum(dim=1, keepdim=True) + (res.float() if fused else 0)
    err = (outs["one_shot"][0].float() - expected).abs().max().item()
    assert err <= 0.02 * expected.abs().max().item(), f"rows={rows}: |one_shot - torch| = {err:.3e}"


@pytest.mark.parametrize("device_params", FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
def test_all_reduce_async_one_shot_chain_in_trace(mesh_device):
    """CHAIN back-to-back fused one_shot all-reduces share one persistent buffer, as a decoder does
    (x_{i+1} = x_i + sum(partials_i), semaphores cycled). One traced replay must equal the same chain in the default
    mode, run eagerly, on every device; a second replay with new inputs must follow them."""
    CHAIN = 32
    rows = 32
    tp, mem, sems, buf = _setup(mesh_device)
    g = torch.Generator().manual_seed(11)
    parts_a = [torch.randn(1, tp, rows, DIM, generator=g).to(torch.bfloat16) for _ in range(CHAIN)]
    parts_b = [(torch.randn(1, tp, rows, DIM, generator=g) * 2.0).to(torch.bfloat16) for _ in range(CHAIN)]
    x0_a = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)
    x0_b = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)
    inps = [_partials_host(mesh_device, p).to(mesh_device, mem) for p in parts_a]
    x0 = _replicated_host(mesh_device, x0_a).to(mesh_device, mem)

    def chain(one_shot):
        x = x0
        for i in range(CHAIN):
            y = _ar(mesh_device, inps[i], buf, sems[i % 2], mem, one_shot, residual=x)
            if not one_shot:
                # The default mode does not order a device's next write into a peer's buffer after that peer's
                # reduction of the previous call; the reference waits for each call instead.
                ttnn.synchronize_device(mesh_device)
            if x is not x0:
                ttnn.deallocate(x)
            x = y
        return x

    def load(parts, x0_t):
        for inp, p in zip(inps, parts):
            ttnn.copy_host_to_device_tensor(_partials_host(mesh_device, p), inp)
        ttnn.copy_host_to_device_tensor(_replicated_host(mesh_device, x0_t), x0)

    refs = {}
    for name, parts, x0_t in (("a", parts_a, x0_a), ("b", parts_b, x0_b)):
        load(parts, x0_t)
        out = chain(one_shot=False)
        refs[name] = _per_device(out)
        ttnn.deallocate(out)
    load(parts_a, x0_a)
    warm = chain(one_shot=True)
    _assert_same(_per_device(warm), refs["a"], "eager one_shot chain")
    ttnn.deallocate(warm)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = chain(one_shot=True)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for rep in range(3):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            _assert_same(_per_device(traced), refs["a"], f"traced chain replay {rep}")
        load(parts_b, x0_b)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        _assert_same(_per_device(traced), refs["b"], "traced chain, new inputs")
    finally:
        ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(traced)


@pytest.mark.parametrize("device_params", FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
def test_all_reduce_async_one_shot_refuses_unsplittable_cores(mesh_device, expect_error):
    """20 output cores (5x4) do not split into 8 owner slices; the default mode still runs on them."""
    tp, mem, sems, buf = _setup(mesh_device, grid_xy=(5, 4))
    full, _ = _random(tp, 8, seed=5)
    inp = _partials_host(mesh_device, full).to(mesh_device, mem)
    with expect_error(RuntimeError, "one_shot splits the output cores"):
        _ar(mesh_device, inp, buf, sems[0], mem, one_shot=True)
    out = _ar(mesh_device, inp, buf, sems[1], mem, one_shot=False)
    ttnn.deallocate(out)


@pytest.mark.parametrize("device_params", [_fabric_params(ttnn.FabricConfig.FABRIC_1D, 16 << 20)], indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
def test_all_reduce_async_one_shot_latency(mesh_device):
    """Traced latency of 64 back-to-back fused all-reduces per replay, default vs one_shot (reported, not a gate)."""
    ITERS, REPLAYS = 64, 20
    tp, mem, sems, buf = _setup(mesh_device)
    full, res = _random(tp, 32, seed=7)
    inp = _partials_host(mesh_device, full).to(mesh_device, mem)
    residual = _replicated_host(mesh_device, res).to(mesh_device, mem)

    def capture(one_shot):
        warm = _ar(mesh_device, inp, buf, sems[0], mem, one_shot, residual)
        ttnn.deallocate(warm)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        prev = None
        for i in range(ITERS):
            out = _ar(mesh_device, inp, buf, sems[i % 2], mem, one_shot, residual)
            if prev is not None:
                ttnn.deallocate(prev)
            prev = out
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        return tid, prev

    def time_us(one_shot):
        # One trace alive at a time: a buffer allocated after a capture and alive at its replay is unsafe.
        tid, last = capture(one_shot)
        try:
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - t0) * 1e6 / (ITERS * REPLAYS)
        finally:
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(last)

    runs = {"default": [], "one_shot": []}
    for _ in range(2):
        for name, one_shot in (("default", False), ("one_shot", True)):
            runs[name].append(time_us(one_shot))
    best = {name: min(v) for name, v in runs.items()}
    logger.info(
        f"all_reduce_async [1,1,32,{DIM}] bf16 + residual, 10x4 cores, 2 links: "
        f"default {best['default']:.2f} us, one_shot {best['one_shot']:.2f} us per call"
    )
