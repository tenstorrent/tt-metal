# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for matmul_reduce_scatter (the implementer must not modify this file).

matmul_reduce_scatter is a multi-device Fabric op, so it cannot run on the single-device `device`
fixture. This module opens its own mesh with the fabric config each placement needs, once per
placement: a module-scoped, parametrized fixture, so pytest groups the tests by placement. It uses
the production router payload (14 KiB + 64 B on Blackhole), as the golden suite does.

Placements:
  "line": the system mesh under FABRIC_2D. Linear along every mesh axis with >= 2 devices.
  "ring": the system mesh under FABRIC_2D_TORUS_XY when the cluster has wrap links (Galaxy).
          Otherwise the box's snake ring opened as 1 x N under FABRIC_2D_TORUS_XY (QuietBox 2 /
          LoudBox), with cluster_axis = 1. Ring and Linear both run there.

Every device holds its own A (..., M, K) and W (K, N). The reference is pure torch in fp32.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import ttnn

from eval.golden_tests.matmul_reduce_scatter.topology import ring_mock_meshes, system_mesh_shape, wrap_axes
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

PCC = {ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}  # keyed by weight dtype (the narrowest input)


# --------------------------------------------------------------------------------------------
# Mesh placements
# --------------------------------------------------------------------------------------------


def _ring_placement():
    """(mesh_shape, fabric_config, cluster_axis) of a real ring on this cluster, or None."""
    if wrap_axes():
        rows, cols = system_mesh_shape()
        axis = next((a for a in wrap_axes() if (rows, cols)[a] >= 3), None)
        return None if axis is None else ((rows, cols), ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis)
    mocks = ring_mock_meshes()
    if not mocks:
        return None
    shape, axis, _ = mocks[0]
    return shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


def _open_mesh(mesh_shape, fabric_config):
    from tests.scripts.common import get_updated_device_params

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        _router_config(),
    )
    params = get_updated_device_params({"fabric_config": fabric_config})
    params.pop("fabric_config")
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **params)


def _close_mesh(mesh):
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def line_mesh():
    mesh = _open_mesh(system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D)
    yield mesh
    _close_mesh(mesh)


@pytest.fixture(scope="module")
def ring_mesh():
    placement = _ring_placement()
    if placement is None:
        pytest.skip("this cluster cannot close a ring")
    shape, fabric_config, axis = placement
    mesh = _open_mesh(shape, fabric_config)
    mesh.ring_cluster_axis = axis
    yield mesh
    _close_mesh(mesh)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _group_size(mesh, cluster_axis):
    return tuple(mesh.shape)[cluster_axis]


def _axes_with_groups(mesh):
    return [a for a in (0, 1) if _group_size(mesh, a) >= 2]


def _max_links(mesh, cluster_axis, ring):
    """Fewest usable links over every hop of every group along cluster_axis."""
    rows, cols = tuple(mesh.shape)
    node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    best = None
    groups = (
        [[(r, c) for r in range(rows)] for c in range(cols)]
        if cluster_axis == 0
        else [[(r, c) for c in range(cols)] for r in range(rows)]
    )
    for grp in groups:
        hops = list(zip(grp[:-1], grp[1:])) + ([(grp[-1], grp[0])] if ring else [])
        for a, b in hops:
            for s, d in ((a, b), (b, a)):
                n = len(ttnn.get_forwarding_link_indices(node(*s), node(*d)))
                best = n if best is None else min(best, n)
    return best or 0


def _split_ok(a_shape, w_shape, scatter_dim, g):
    extent = a_shape[-2] if scatter_dim == -2 else w_shape[-1]
    return extent % (32 * g) == 0


def _stacked(mesh, shape, scale=1.0):
    rows, cols = tuple(mesh.shape)
    return (torch.randn((rows, cols, *shape), dtype=torch.float32) * scale).to(torch.bfloat16)


def _to_mesh(stacked, mesh, dtype):
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )


def _quantize(stacked, dtype):
    """The weight as the device holds it (block-float rounding), so PCC scores the op, not the format."""
    if dtype == ttnn.bfloat8_b:
        flat = stacked.reshape(-1, stacked.shape[-1])
        t = ttnn.from_torch(flat, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)
    return stacked


def torch_reference(a_stacked, w_stacked, cluster_axis, scatter_dim):
    """Stacked A (R, C, ..., M, K), W (R, C, K, N) -> stacked per-device outputs (fp32)."""
    rows, cols = a_stacked.shape[0], a_stacked.shape[1]
    w = w_stacked.float().reshape(rows, cols, *([1] * (a_stacked.dim() - 4)), *w_stacked.shape[2:])
    partial = torch.matmul(a_stacked.float(), w)
    total = partial.sum(dim=cluster_axis, keepdim=True).expand_as(partial)
    g = (rows, cols)[cluster_axis]
    out = [
        [torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[cluster_axis]] for c in range(cols)] for r in range(rows)
    ]
    return out


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den == 0 else float((a * b).sum() / den)


def _expected_shape(a_shape, w_shape, scatter_dim, g):
    out = [*a_shape[:-1], w_shape[-1]]
    out[scatter_dim] //= g
    return out


def _run_and_check(
    mesh,
    a_shape,
    w_shape,
    *,
    cluster_axis,
    scatter_dim,
    topology,
    num_links,
    weight_dtype,
    compute_kernel_config=None,
    seed=42,
):
    g = _group_size(mesh, cluster_axis)
    if not _split_ok(a_shape, w_shape, scatter_dim, g):
        pytest.skip(f"scattered dim does not split into {g} tile blocks")
    ring = topology == ttnn.Topology.Ring
    if num_links > _max_links(mesh, cluster_axis, ring):
        pytest.skip(f"num_links={num_links} exceeds the links this cluster has per hop")
    torch.manual_seed(seed)
    a = _stacked(mesh, a_shape)
    w = _quantize(_stacked(mesh, w_shape, scale=w_shape[0] ** -0.5), weight_dtype)
    expected = torch_reference(a, w, cluster_axis, scatter_dim)

    kwargs = {} if compute_kernel_config is None else {"compute_kernel_config": compute_kernel_config}
    out = matmul_reduce_scatter(
        _to_mesh(a, mesh, ttnn.bfloat16),
        _to_mesh(w, mesh, weight_dtype),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=num_links,
        **kwargs,
    )
    assert list(out.shape) == _expected_shape(a_shape, w_shape, scatter_dim, g)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    cols = tuple(mesh.shape)[1]
    worst = 1.0
    for idx, dev in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(dev).float()
        assert not torch.isnan(actual).any() and not torch.isinf(actual).any(), f"device ({r},{c}): nan/inf"
        worst = min(worst, _pcc(actual, expected[r][c]))
    assert worst >= PCC[weight_dtype], f"worst-device PCC {worst:.6f} < {PCC[weight_dtype]}"
    return out


# Per-device (A, W) shapes. "one_tile_block" is built per group size: every block is a single tile row/column.
SHAPES = {
    "multi_tile": ((1, 1, 256, 512), (512, 1024)),
    "non_square": ((1, 1, 640, 1024), (1024, 2048)),
    "rank3": ((1, 512, 256), (256, 768)),
    "rank2": ((256, 384), (384, 512)),
    "perf_focus": ((1, 1, 640, 2048), (2048, 7168)),
}


def _shape(name, g):
    if name == "one_tile_block":
        return (1, 1, 32 * g, 64), (64, 32 * g)
    return SHAPES[name]


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["w_bf16", "w_bf8b"])
@pytest.mark.parametrize("scatter_dim", [-2, -1])
@pytest.mark.parametrize("shape_name", ["one_tile_block", *SHAPES])
def test_linear(line_mesh, shape_name, scatter_dim, weight_dtype):
    """Linear along every mesh axis that has a group, 1 link."""
    axes = _axes_with_groups(line_mesh)
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    for cluster_axis in axes:
        a_shape, w_shape = _shape(shape_name, _group_size(line_mesh, cluster_axis))
        if not _split_ok(a_shape, w_shape, scatter_dim, _group_size(line_mesh, cluster_axis)):
            continue
        _run_and_check(
            line_mesh,
            a_shape,
            w_shape,
            cluster_axis=cluster_axis,
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Linear,
            num_links=1,
            weight_dtype=weight_dtype,
        )


@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("scatter_dim", [-2, -1])
def test_linear_links(line_mesh, scatter_dim, num_links):
    """num_links is honoured (1 and 2 links per hop), on the PERF FOCUS shape with bf8b weights."""
    a_shape, w_shape = SHAPES["perf_focus"]
    for cluster_axis in _axes_with_groups(line_mesh):
        if not _split_ok(a_shape, w_shape, scatter_dim, _group_size(line_mesh, cluster_axis)):
            continue
        _run_and_check(
            line_mesh,
            a_shape,
            w_shape,
            cluster_axis=cluster_axis,
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            weight_dtype=ttnn.bfloat8_b,
        )


@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear], ids=["ring", "linear"])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("scatter_dim", [-2, -1])
@pytest.mark.parametrize("shape_name", ["one_tile_block", "multi_tile", "perf_focus"])
def test_ring(ring_mesh, shape_name, scatter_dim, num_links, topology):
    """Both ring directions (balanced on an even group) and Linear under a torus config."""
    cluster_axis = ring_mesh.ring_cluster_axis
    g = _group_size(ring_mesh, cluster_axis)
    a_shape, w_shape = _shape(shape_name, g)
    _run_and_check(
        ring_mesh,
        a_shape,
        w_shape,
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=num_links,
        weight_dtype=ttnn.bfloat8_b,
    )
    if topology == ttnn.Topology.Ring:
        plan = importlib.import_module("ttnn.operations.matmul_reduce_scatter")._LAST_PLAN
        assert plan["schedule"] == ("ring_balanced" if g % 2 == 0 else "ring_unbalanced"), plan


@pytest.mark.parametrize("residency", [True, False], ids=["resident", "stream"])
@pytest.mark.parametrize("scatter_dim", [-2, -1])
def test_regime_pinned(line_mesh, monkeypatch, scatter_dim, residency):
    """Both cross-block-operand regimes, pinned through the package hook (independent of the grid size)."""
    pkg = importlib.import_module("ttnn.operations.matmul_reduce_scatter")
    monkeypatch.setattr(pkg, "_FORCE_XBLOCK_RESIDENCY", residency)
    a_shape, w_shape = SHAPES["multi_tile"]
    for cluster_axis in _axes_with_groups(line_mesh):
        if not _split_ok(a_shape, w_shape, scatter_dim, _group_size(line_mesh, cluster_axis)):
            continue
        _run_and_check(
            line_mesh,
            a_shape,
            w_shape,
            cluster_axis=cluster_axis,
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Linear,
            num_links=1,
            weight_dtype=ttnn.bfloat16,
        )
        assert pkg._LAST_PLAN["xblock"] == ("resident" if residency else "stream"), pkg._LAST_PLAN


def test_regime_natural(line_mesh):
    """Natural selection: the PERF FOCUS shape keeps A resident (scatter -1). The MiMo shape with a bf16
    weight cannot keep W resident (scatter -2), so it streams."""
    pkg = importlib.import_module("ttnn.operations.matmul_reduce_scatter")
    axes = _axes_with_groups(line_mesh)
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    cluster_axis = axes[-1]
    a_shape, w_shape = SHAPES["perf_focus"]
    _run_and_check(
        line_mesh,
        a_shape,
        w_shape,
        cluster_axis=cluster_axis,
        scatter_dim=-1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        weight_dtype=ttnn.bfloat8_b,
    )
    assert pkg._LAST_PLAN["xblock"] == "resident", pkg._LAST_PLAN
    a_shape, w_shape = (1, 1, 2048, 2048), (2048, 4096)
    _run_and_check(
        line_mesh,
        a_shape,
        w_shape,
        cluster_axis=cluster_axis,
        scatter_dim=-2,
        topology=ttnn.Topology.Linear,
        num_links=1,
        weight_dtype=ttnn.bfloat16,
    )
    assert pkg._LAST_PLAN["xblock"] == "stream", pkg._LAST_PLAN


@pytest.mark.parametrize("scatter_dim", [-2, -1])
def test_every_partial_exactly_once(line_mesh, scatter_dim):
    """A = ones on the first 32 K columns; W[r, c] = constant rank (r*C + c + 1) on those rows. Every partial
    element is exactly 32*rank and every relay / final sum is an exact bf16 integer. A dropped, doubled or
    foreign partial is a wrong integer."""
    rows, cols = tuple(line_mesh.shape)
    a_shape, w_shape = (1, 1, 256, 64), (64, 512)
    for cluster_axis in _axes_with_groups(line_mesh):
        g = _group_size(line_mesh, cluster_axis)
        if not _split_ok(a_shape, w_shape, scatter_dim, g):
            continue
        a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
        a[..., :32] = 1
        ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols, 1, 1)
        w = torch.zeros((rows, cols, *w_shape), dtype=torch.float32)
        w[:, :, :32, :] = ranks
        w = w.to(torch.bfloat16)
        expected = torch_reference(a, w, cluster_axis, scatter_dim)
        out = matmul_reduce_scatter(
            _to_mesh(a, line_mesh, ttnn.bfloat16),
            _to_mesh(w, line_mesh, ttnn.bfloat16),
            cluster_axis=cluster_axis,
            scatter_dim=scatter_dim,
            topology=ttnn.Topology.Linear,
            num_links=1,
        )
        for idx, dev in enumerate(ttnn.get_device_tensors(out)):
            r, c = divmod(idx, cols)
            actual = ttnn.to_torch(dev).float()
            bad = (actual != expected[r][c]).sum().item()
            assert bad == 0, f"axis {cluster_axis} device ({r},{c}): {bad}/{actual.numel()} elements differ"


def test_repeated_calls_deterministic(line_mesh):
    """Program-cache reuse with new tensors (only runtime args change), no stale scratch or counters:
    new inputs give correct results, and repeating an input gives a bit-identical result."""
    axes = _axes_with_groups(line_mesh)
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    cluster_axis = axes[-1]
    a_shape, w_shape = SHAPES["multi_tile"]
    first = None
    for seed in (42, 7, 42, 42):
        out = _run_and_check(
            line_mesh,
            a_shape,
            w_shape,
            cluster_axis=cluster_axis,
            scatter_dim=-1,
            topology=ttnn.Topology.Linear,
            num_links=1,
            weight_dtype=ttnn.bfloat16,
            seed=seed,
        )
        if seed == 42:
            host = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
            if first is None:
                first = host
            else:
                for x, y in zip(first, host):
                    assert torch.equal(x, y), "output differs between identical calls"


def test_compute_kernel_config(line_mesh):
    """Explicit maxed precision (HiFi4, fp32 DEST), and the production 16-bit floor (HiFi2, no fp32 DEST)."""
    axes = _axes_with_groups(line_mesh)
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    a_shape, w_shape = SHAPES["non_square"]
    for cfg in (
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
    ):
        _run_and_check(
            line_mesh,
            a_shape,
            w_shape,
            cluster_axis=axes[-1],
            scatter_dim=-1,
            topology=ttnn.Topology.Linear,
            num_links=1,
            weight_dtype=ttnn.bfloat16,
            compute_kernel_config=cfg,
        )


def test_caller_errors(line_mesh, expect_error):
    """Caller errors are ValueError, not support refusals."""
    axes = _axes_with_groups(line_mesh)
    if not axes:
        pytest.skip("no mesh axis has >= 2 devices")
    cluster_axis = axes[-1]
    g = _group_size(line_mesh, cluster_axis)
    torch.manual_seed(42)
    a = _to_mesh(_stacked(line_mesh, (1, 1, 32 * g, 64)), line_mesh, ttnn.bfloat16)
    w_bad_k = _to_mesh(_stacked(line_mesh, (96, 32 * g)), line_mesh, ttnn.bfloat16)
    with expect_error(ValueError, ""):
        matmul_reduce_scatter(a, w_bad_k, cluster_axis=cluster_axis)
    w = _to_mesh(_stacked(line_mesh, (64, 32 * g + 32)), line_mesh, ttnn.bfloat16)
    with expect_error(ValueError, ""):  # N does not split into G tile blocks
        matmul_reduce_scatter(a, w, cluster_axis=cluster_axis, scatter_dim=-1)
    w = _to_mesh(_stacked(line_mesh, (64, 32 * g)), line_mesh, ttnn.bfloat16)
    if _max_links(line_mesh, cluster_axis, ring=False) < 2:
        with expect_error(ValueError, ""):  # more links than a hop has (2 is a SUPPORTED value, so validate passes)
            matmul_reduce_scatter(a, w, cluster_axis=cluster_axis, num_links=2)
    if g >= 3:  # a 2-device "ring" is the line; a Ring of >= 3 needs a torus config (this mesh is FABRIC_2D)
        with expect_error(ValueError, ""):
            matmul_reduce_scatter(a, w, cluster_axis=cluster_axis, topology=ttnn.Topology.Ring)
