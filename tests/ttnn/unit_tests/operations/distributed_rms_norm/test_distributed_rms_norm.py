# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for distributed_rms_norm (immutable spec — do not edit).

A multi-device op: every device (r, c) holds x[r, c] of per-device shape (..., M, W_local); devices along
`cluster_axis` form a group whose slices, in group-position order, are the full hidden dim W = G * W_local.
Reference: torch.nn.functional.rms_norm over the concatenated row, sliced back per device.

Meshes are opened here with FABRIC_2D_TORUS_XY and the production router payload (the golden suite's
configuration), cached per mesh shape: the system mesh for Linear groups, and the box's snake ring (1 x N / N x 1,
topology.ring_mock_meshes) or a real wrap axis for Ring groups. The root `device` fixture is not used (single-device).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.topology import (
    ring_mock_meshes,
    system_mesh_shape,
    topology_feasible,
    wrap_axes,
)
from eval.sharding import auto_shard_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

torch.manual_seed(42)

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
EPS = 1e-6


# ---------------------------------------------------------------------------------------------------------------
# Mesh handling
# ---------------------------------------------------------------------------------------------------------------

_SYSTEM = system_mesh_shape()
_MULTI = _SYSTEM[0] * _SYSTEM[1] > 1
pytestmark = pytest.mark.skipif(not _MULTI, reason="distributed_rms_norm needs a multi-device system")


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


def _linear_meshes():
    """(mesh_shape, cluster_axis) pairs with a group of >= 2 on the system mesh."""
    return [(_SYSTEM, ax) for ax in (0, 1) if _SYSTEM[ax] >= 2]


def _ring_meshes():
    """(mesh_shape, cluster_axis) pairs where Ring is a real ring: a wrap axis of the system mesh, or the box's
    snake ring opened as 1 x N / N x 1."""
    out = [(_SYSTEM, ax) for ax in (0, 1) if topology_feasible(ttnn.Topology.Ring, ax, _SYSTEM) is None]
    if not out:
        out = list(ring_mock_meshes())
    return out


def _links_available(mesh, cluster_axis, ring):
    """Minimum number of forwarding links over every hop of every group along cluster_axis."""
    rows, cols = tuple(mesh.shape)
    node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    groups = (
        [[(r, c) for r in range(rows)] for c in range(cols)]
        if cluster_axis == 0
        else [[(r, c) for c in range(cols)] for r in range(rows)]
    )
    best = None
    for grp in groups:
        G = len(grp)
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring and G >= 3:
            hops.append((grp[-1], grp[0]))
        for a, b in hops:
            for s, d in ((a, b), (b, a)):
                n = len(ttnn.get_forwarding_link_indices(node(*s), node(*d)))
                best = n if best is None else min(best, n)
    return best or 0


# ---------------------------------------------------------------------------------------------------------------
# Tensors and reference
# ---------------------------------------------------------------------------------------------------------------


def _stacked(mesh_shape, shape, torch_dtype, scale=1.0):
    rows, cols = mesh_shape
    return (torch.randn((rows, cols, *shape), dtype=torch.float32) * scale).to(torch_dtype)


def _to_mesh(stacked, mesh, *, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    rows, cols = stacked.shape[0], stacked.shape[1]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _gamma_to_mesh(gamma_stacked, mesh, *, dtype, layout):
    """Logical (R, C, 1, 1, 1, W_local) -> TILE (1, 1, 1, W_local) or ROW_MAJOR sticks (1, 1, ceil(W/32), 32)."""
    g = gamma_stacked
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        w = g.shape[-1]
        padded = -(-w // 32) * 32
        if padded != w:
            g = torch.nn.functional.pad(g, (0, padded - w))
        g = g.reshape(*g.shape[:-2], padded // 32, 32)
    return _to_mesh(g, mesh, dtype=dtype, layout=layout)


def _quantize(stacked, dtype):
    if dtype != ttnn.bfloat8_b:
        return stacked
    flat = stacked.reshape(-1, stacked.shape[-1])
    t = ttnn.from_torch(flat, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)


def _reference(x_stacked, gamma_stacked, cluster_axis, eps):
    x = x_stacked.to(torch.float32)
    g = None if gamma_stacked is None else gamma_stacked.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    w_local = x.shape[-1]
    out = torch.empty_like(x)
    for other in range(cols if cluster_axis == 0 else rows):
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None if g is None else torch.cat([g[r, c].reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=eps)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def _pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    ac, bc = a - a.mean(), b - b.mean()
    den = ac.norm() * bc.norm()
    return 1.0 if den < 1e-30 else float((ac * bc).sum() / den)


def _check(out, expected, *, shape, dtype, mesh_shape):
    assert list(out.shape) == list(shape), f"shape {list(out.shape)} != {list(shape)}"
    assert out.dtype == dtype
    assert out.layout == ttnn.TILE_LAYOUT
    devs = ttnn.get_device_tensors(out)
    rows, cols = mesh_shape
    assert len(devs) == rows * cols
    for idx, d in enumerate(devs):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(d).to(torch.float32)
        assert torch.isfinite(got).all(), f"device ({r},{c}) has non-finite output"
        pcc = _pcc(got, expected[r, c])
        assert pcc >= PCC[dtype], f"device ({r},{c}) PCC {pcc:.6f} < {PCC[dtype]}"


def _run(
    mesh,
    shape,
    *,
    cluster_axis,
    topology,
    dtype=ttnn.bfloat16,
    gamma_layout=ttnn.TILE_LAYOUT,
    gamma_dtype=ttnn.bfloat16,
    num_links=None,
    fp32_dest_acc_en=True,
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    eps=EPS,
):
    mesh_shape = tuple(mesh.shape)
    x = _quantize(_stacked(mesh_shape, shape, TORCH_DTYPE[dtype]), dtype)
    gamma = None
    tg = None
    if gamma_layout is not None:
        gamma = _stacked(mesh_shape, (1, 1, 1, shape[-1]), TORCH_DTYPE[gamma_dtype])
        tg = _gamma_to_mesh(gamma, mesh, dtype=gamma_dtype, layout=gamma_layout)
    kwargs = {}
    if memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=mesh)
        kwargs["memory_config"] = mc
    tx = _to_mesh(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )
    out = distributed_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=eps,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=cfg,
        **kwargs,
    )
    expected = _reference(x, gamma, cluster_axis, eps)
    _check(out, expected, shape=shape, dtype=dtype, mesh_shape=mesh_shape)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Linear groups on the system mesh
# ---------------------------------------------------------------------------------------------------------------

LINEAR_SHAPES = [
    (1, 1, 32, 32),  # single tile (regime D1)
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 32, 64),  # Wt < MIN_SLICE_TILES: regime D1 pin
    (1, 1, 640, 1792),  # perf focus geometry: regime D3, several waves
    (1, 1, 32, 1792),  # decode: D3, one wave
    (2, 4, 128, 512),  # multi-batch
    (1, 32, 1792),  # rank 3
    (640, 1792),  # rank 2
    (1, 1, 17, 50),  # both non-aligned
    (1, 1, 100, 512),  # H non-aligned
    (1, 1, 64, 1000),  # W non-aligned, multi-tile
]

_LINEAR = [
    pytest.param(ms, ax, shape, dtype, gl, id=f"{ms[0]}x{ms[1]}-ax{ax}-{'x'.join(map(str, shape))}-{dname}-{gname}")
    for ms, ax in _linear_meshes()
    for shape in LINEAR_SHAPES
    for dtype, dname in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32"))
    for gl, gname in ((ttnn.TILE_LAYOUT, "gTILE"), (ttnn.ROW_MAJOR_LAYOUT, "gRM"), (None, "nog"))
]


@pytest.mark.parametrize("mesh_shape,cluster_axis,shape,dtype,gamma_layout", _LINEAR)
def test_linear(mesh_cache, mesh_shape, cluster_axis, shape, dtype, gamma_layout):
    mesh = mesh_cache.get(mesh_shape)
    _run(mesh, shape, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear, dtype=dtype, gamma_layout=gamma_layout)


@pytest.mark.parametrize(
    "mesh_shape,cluster_axis", [pytest.param(ms, ax, id=f"{ms[0]}x{ms[1]}-ax{ax}") for ms, ax in _linear_meshes()]
)
@pytest.mark.parametrize(
    "dtype,fp32_dest_acc_en,gamma_dtype,shape",
    [
        pytest.param(ttnn.bfloat16, False, ttnn.bfloat16, (1, 1, 640, 1792), id="bf16-dest16"),
        pytest.param(ttnn.bfloat16, True, ttnn.float32, (1, 1, 64, 1000), id="bf16-gfp32"),
        pytest.param(ttnn.float32, True, ttnn.bfloat16, (1, 1, 32, 1792), id="fp32-gbf16"),
        pytest.param(ttnn.bfloat8_b, True, ttnn.bfloat16, (1, 1, 128, 1024), id="bf8b"),
        pytest.param(ttnn.bfloat8_b, False, ttnn.bfloat16, (1, 1, 64, 1000), id="bf8b-dest16-wnonaligned"),
    ],
)
def test_numeric_configs(mesh_cache, mesh_shape, cluster_axis, dtype, fp32_dest_acc_en, gamma_dtype, shape):
    mesh = mesh_cache.get(mesh_shape)
    _run(
        mesh,
        shape,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        dtype=dtype,
        gamma_dtype=gamma_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )


@pytest.mark.parametrize(
    "mesh_shape,cluster_axis", [pytest.param(ms, ax, id=f"{ms[0]}x{ms[1]}-ax{ax}") for ms, ax in _linear_meshes()]
)
@pytest.mark.parametrize(
    "memory_layout,shape",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1, 640, 1792), id="height-640x1792"),
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1, 2048, 1280), id="height-2048x1280"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 32, 1792), id="width-32x1792"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 640, 1792), id="width-640x1792"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 64, 1000), id="width-wnonaligned"),
    ],
)
def test_sharded(mesh_cache, mesh_shape, cluster_axis, memory_layout, shape):
    mesh = mesh_cache.get(mesh_shape)
    out = _run(mesh, shape, cluster_axis=cluster_axis, topology=ttnn.Topology.Linear, memory_layout=memory_layout)
    assert out.memory_config().memory_layout == memory_layout


@pytest.mark.parametrize(
    "mesh_shape,cluster_axis", [pytest.param(ms, ax, id=f"{ms[0]}x{ms[1]}-ax{ax}") for ms, ax in _linear_meshes()]
)
def test_back_to_back_deterministic(mesh_cache, mesh_shape, cluster_axis):
    """Several calls with no host synchronization in between (the ready fence and counter re-arm), then bit-identical
    results run to run."""
    mesh = mesh_cache.get(mesh_shape)
    shape = (1, 1, 640, 1792)
    x = _stacked(tuple(mesh.shape), shape, torch.bfloat16)
    g = _stacked(tuple(mesh.shape), (1, 1, 1, shape[-1]), torch.bfloat16)
    tx = _to_mesh(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = _gamma_to_mesh(g, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    outs = [
        distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=EPS, topology=ttnn.Topology.Linear)
        for _ in range(4)
    ]
    expected = _reference(x, g, cluster_axis, EPS)
    _check(outs[-1], expected, shape=shape, dtype=ttnn.bfloat16, mesh_shape=tuple(mesh.shape))
    first = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(outs[0])]
    for o in outs[1:]:
        for a, b in zip(first, [ttnn.to_torch(d) for d in ttnn.get_device_tensors(o)]):
            assert torch.equal(a, b), "results differ run to run"


# ---------------------------------------------------------------------------------------------------------------
# Ring groups (real wrap axis, or the box's snake ring) — the production topology
# ---------------------------------------------------------------------------------------------------------------

RING_SHAPES = [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280), (1, 1, 17, 50)]

_RING = [
    pytest.param(ms, ax, topo, nl, shape, id=f"{ms[0]}x{ms[1]}-ax{ax}-{topo.name}-L{nl}-{'x'.join(map(str, shape))}")
    for ms, ax in _ring_meshes()
    for topo in (ttnn.Topology.Ring, ttnn.Topology.Linear)
    for nl in (1, 2)
    for shape in RING_SHAPES
]


@pytest.mark.skipif(not _RING, reason="no ring of >= 3 devices on this cluster")
@pytest.mark.parametrize("mesh_shape,cluster_axis,topology,num_links,shape", _RING)
def test_ring(mesh_cache, mesh_shape, cluster_axis, topology, num_links, shape):
    mesh = mesh_cache.get(mesh_shape)
    if _links_available(mesh, cluster_axis, topology == ttnn.Topology.Ring) < num_links:
        pytest.skip(f"fewer than {num_links} links on some hop")
    _run(mesh, shape, cluster_axis=cluster_axis, topology=topology, num_links=num_links, gamma_layout=ttnn.TILE_LAYOUT)


@pytest.mark.skipif(not _RING, reason="no ring of >= 3 devices on this cluster")
@pytest.mark.parametrize(
    "mesh_shape,cluster_axis", [pytest.param(ms, ax, id=f"{ms[0]}x{ms[1]}-ax{ax}") for ms, ax in _ring_meshes()]
)
def test_ring_gamma_rm_fp32_and_default_links(mesh_cache, mesh_shape, cluster_axis):
    mesh = mesh_cache.get(mesh_shape)
    _run(
        mesh,
        (1, 1, 640, 1792),
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Ring,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_dtype=ttnn.float32,
        num_links=None,
        eps=1e-5,
    )


# ---------------------------------------------------------------------------------------------------------------
# Validation (caller errors, not support refusals)
# ---------------------------------------------------------------------------------------------------------------


def _any_mesh_and_axis():
    lm = _linear_meshes()
    return lm[0] if lm else (None, None)


def test_rejects_rank_1(mesh_cache, expect_error):
    ms, ax = _any_mesh_and_axis()
    mesh = mesh_cache.get(ms)
    rows, cols = tuple(mesh.shape)
    t = ttnn.from_torch(
        torch.randn(rows * cols * 64),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(t, cluster_axis=ax)


def test_rejects_bad_gamma_shape(mesh_cache, expect_error):
    ms, ax = _any_mesh_and_axis()
    mesh = mesh_cache.get(ms)
    shape = (1, 1, 32, 128)
    x = _to_mesh(_stacked(tuple(mesh.shape), shape, torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    bad = _to_mesh(
        _stacked(tuple(mesh.shape), (1, 1, 1, 64), torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax, gamma=bad)
    bad_rm = _to_mesh(
        _stacked(tuple(mesh.shape), (1, 1, 1, 128), torch.bfloat16),
        mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )  # a ROW_MAJOR gamma must be sticks (1, 1, 4, 32), not one row
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax, gamma=bad_rm)


def test_rejects_too_many_links(mesh_cache, expect_error):
    ms, ax = _any_mesh_and_axis()
    mesh = mesh_cache.get(ms)
    avail = _links_available(mesh, ax, ring=False)
    shape = (1, 1, 32, 128)
    x = _to_mesh(_stacked(tuple(mesh.shape), shape, torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax, num_links=avail + 1)
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax, num_links=0)


def test_rejects_axis_of_size_one(mesh_cache, expect_error):
    """A mesh axis of size 1 has nothing to reduce: on the snake-ring mesh (1 x N) cluster_axis 0 has size 1."""
    cands = [ms for ms, _ in ring_mock_meshes()] + [_SYSTEM]
    target = next(((ms, ax) for ms in cands for ax in (0, 1) if ms[ax] == 1 and ms[1 - ax] >= 2), None)
    if target is None:
        pytest.skip("no mesh with an axis of size 1 on this cluster")
    ms, ax = target
    mesh = mesh_cache.get(ms)
    x = _to_mesh(
        _stacked(tuple(mesh.shape), (1, 1, 32, 128), torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax)


def test_rejects_ring_without_wrap(mesh_cache, expect_error):
    """Ring along an axis of >= 3 devices without a wrap link is a caller error (skipped where no such axis exists)."""
    target = next(
        ((_SYSTEM, ax) for ax in (0, 1) if _SYSTEM[ax] >= 3 and ax not in wrap_axes()),
        None,
    )
    if target is None:
        pytest.skip("no axis of >= 3 devices without a wrap link on this cluster")
    ms, ax = target
    mesh = mesh_cache.get(ms)
    x = _to_mesh(
        _stacked(tuple(mesh.shape), (1, 1, 32, 128), torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), ""):
        distributed_rms_norm(x, cluster_axis=ax, topology=ttnn.Topology.Ring)
