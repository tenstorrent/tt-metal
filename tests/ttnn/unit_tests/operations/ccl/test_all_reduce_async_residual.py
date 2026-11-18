# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_reduce_async(residual_tensor=...): the residual add inside the reduction kernel.

The shape is the Qwen3.8-27B TP=8 decode residual: [1, 1, rows, 5120] bf16, L1 width-sharded over a 10x4 core grid
(shard [32, 128]), with a persistent [32, 5120 * 8] buffer on the same grid.

Run on 8 chips:
  pytest tests/ttnn/unit_tests/operations/ccl/test_all_reduce_async_residual.py
"""

import pytest
import torch

import ttnn

GRID = (10, 4)
DIM = 5120


def _fabric_params(fabric_config):
    return {"fabric_config": fabric_config, "trace_region_size": 8 << 20}


FABRICS = [
    pytest.param(_fabric_params(ttnn.FabricConfig.FABRIC_1D), id="fabric_1d"),
    pytest.param(_fabric_params(ttnn.FabricConfig.FABRIC_1D_RING), id="fabric_1d_ring"),
]
MESH_1X8 = [pytest.param((1, 8), id="1x8")]


def _width_mem(shard_w, grid_xy=GRID):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_xy[0] - 1, grid_xy[1] - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _setup(mesh_device):
    tp = list(mesh_device.shape)[1]
    ncores = GRID[0] * GRID[1]
    mem = _width_mem(DIM // ncores)
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(2)]
    buf = ttnn.from_torch(
        torch.zeros([1, 1, ttnn.TILE_SIZE, DIM * tp]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=_width_mem(DIM * tp // ncores),
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


def _random(tp, rows, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn(1, tp, rows, DIM, generator=g) * scale).to(torch.bfloat16)
    res = (torch.randn(1, 1, rows, DIM, generator=g) * 4.0).to(torch.bfloat16)
    return full, res


def _ar(mesh_device, inp, buf, sem, mem, residual=None):
    kwargs = {} if residual is None else {"residual_tensor": residual}
    return ttnn.experimental.all_reduce_async(
        inp,
        buf,
        cluster_axis=1,
        mesh_device=mesh_device,
        multi_device_global_semaphore=sem,
        topology=ttnn.Topology.Ring,
        num_links=2,
        memory_config=mem,
        **kwargs,
    )


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _separate_add(mesh_device, inp, residual, buf, sem, mem):
    """The unfused reference: plain all-reduce, then ttnn.add(residual, sum) in the same layout."""
    ar = _ar(mesh_device, inp, buf, sem, mem)
    out = ttnn.add(residual, ar, memory_config=mem)
    ttnn.deallocate(ar)
    ref = _per_device(out)
    ttnn.deallocate(out)
    return ref


def _assert_same(got, ref, what):
    for dev, (g, r) in enumerate(zip(got, ref)):
        assert g.shape == r.shape, f"{what}, device {dev}: shape {tuple(g.shape)} vs {tuple(r.shape)}"
        assert torch.equal(g, r), (
            f"{what}, device {dev}: max |diff| {(g.float() - r.float()).abs().max().item():.3e}, "
            f"{(g != r).sum().item()} elements differ"
        )


@pytest.mark.parametrize("device_params", FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_all_reduce_async_residual_matches_separate_add(mesh_device, rows):
    """Fused == all-reduce then ttnn.add, byte for byte, on every device and on program-cache hits."""
    tp, mem, sems, buf = _setup(mesh_device)
    full, res = _random(tp, rows, seed=rows)
    inp = _partials_host(mesh_device, full).to(mesh_device, mem)
    residual = _replicated_host(mesh_device, res).to(mesh_device, mem)

    ref = _separate_add(mesh_device, inp, residual, buf, sems[0], mem)
    for dev in range(1, tp):
        assert torch.equal(ref[dev], ref[0]), f"separate add: device {dev} differs from device 0"
    for it in range(3):
        out = _ar(mesh_device, inp, buf, sems[(it + 1) % 2], mem, residual=residual)
        got = _per_device(out)
        ttnn.deallocate(out)
        _assert_same(got, ref, f"rows={rows} call {it}")

    expected = full.float().sum(dim=1, keepdim=True) + res.float()
    err = (got[0].float() - expected).abs().max().item()
    assert err <= 0.02 * expected.abs().max().item(), f"rows={rows}: |fused - torch| = {err:.3e}"
    for dev, r in enumerate(_per_device(residual)):
        assert torch.equal(r, res), f"device {dev}: the residual tensor changed"


@pytest.mark.parametrize("device_params", FABRICS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
def test_all_reduce_async_residual_trace_follows_buffer_contents(mesh_device):
    """A captured fused all-reduce reads the input and residual buffers at replay: new contents, new result."""
    tp, mem, sems, buf = _setup(mesh_device)
    rows = 32
    full_a, res_a = _random(tp, rows, seed=1)
    full_b, res_b = _random(tp, rows, seed=2, scale=3.0)
    inp = _partials_host(mesh_device, full_a).to(mesh_device, mem)
    residual = _replicated_host(mesh_device, res_a).to(mesh_device, mem)

    ref_a = _separate_add(mesh_device, inp, residual, buf, sems[0], mem)
    ttnn.copy_host_to_device_tensor(_partials_host(mesh_device, full_b), inp)
    ttnn.copy_host_to_device_tensor(_replicated_host(mesh_device, res_b), residual)
    ref_b = _separate_add(mesh_device, inp, residual, buf, sems[0], mem)
    ttnn.copy_host_to_device_tensor(_partials_host(mesh_device, full_a), inp)
    ttnn.copy_host_to_device_tensor(_replicated_host(mesh_device, res_a), residual)
    warm = _ar(mesh_device, inp, buf, sems[1], mem, residual=residual)
    ttnn.deallocate(warm)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = _ar(mesh_device, inp, buf, sems[1], mem, residual=residual)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for rep in range(2):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            _assert_same(_per_device(traced), ref_a, f"replay {rep}, inputs A")
        ttnn.copy_host_to_device_tensor(_partials_host(mesh_device, full_b), inp)
        ttnn.copy_host_to_device_tensor(_replicated_host(mesh_device, res_b), residual)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        _assert_same(_per_device(traced), ref_b, "replay, inputs B")
    finally:
        ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(traced)


@pytest.mark.parametrize("device_params", FABRICS[:1], indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_1X8, indirect=True)
@pytest.mark.parametrize("case", ["dram_interleaved", "other_shard_spec", "bfloat8_b"])
def test_all_reduce_async_residual_refuses_bad_residual(mesh_device, case, expect_error):
    tp, mem, sems, buf = _setup(mesh_device)
    rows = 8
    full, res = _random(tp, rows, seed=3)
    inp = _partials_host(mesh_device, full).to(mesh_device, mem)
    if case == "dram_interleaved":
        residual = _replicated_host(mesh_device, res).to(mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    elif case == "other_shard_spec":
        residual = _replicated_host(mesh_device, res).to(mesh_device, _width_mem(DIM // 32, (8, 4)))
    else:
        residual = ttnn.from_torch(
            res,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    with expect_error(RuntimeError, "Fused residual"):
        _ar(mesh_device, inp, buf, sems[0], mem, residual=residual)
