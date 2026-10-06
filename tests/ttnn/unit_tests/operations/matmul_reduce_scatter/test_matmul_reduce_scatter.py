# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for matmul_reduce_scatter (immutable spec — do not modify).

Fused multi-device matmul + reduce-scatter (SUM): every device (r, c) holds A[r, c] (..., M, K) and
W[r, c] (K, N); the group along `cluster_axis` (G devices, position p) sums A[g] @ W[g] and device p keeps
block p of the sum along `scatter_dim` (-2: rows, -1: columns). Output is bfloat16, TILE, per device.

This is a multi-device CCL op, so the single-device `device` fixture cannot run it. The module opens the
whole system mesh ONCE with FABRIC_2D (the config the golden suite uses for Linear) in a module-scoped
`mesh_device` fixture and closes it at the end of the module.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter.py
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

# PCC keyed by the weight dtype (the lowest-precision operand).
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


# --------------------------------------------------------------------------------------------------------------------
# Mesh fixture
# --------------------------------------------------------------------------------------------------------------------


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


@pytest.fixture(scope="module")
def mesh_device():
    from tests.scripts.common import get_updated_device_params

    shape = _system_mesh_shape()
    if len(shape) != 2 or shape[0] * shape[1] < 2:
        pytest.skip(f"matmul_reduce_scatter needs a 2-D mesh with >= 2 devices, system mesh is {shape}")
    fabric = ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)
    params = get_updated_device_params({"fabric_config": fabric})
    params.pop("fabric_config")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _group_size(mesh_device, cluster_axis):
    return tuple(mesh_device.shape)[cluster_axis]


def _usable_links(mesh_device, cluster_axis):
    """Min over every hop of every group along cluster_axis of the links the active fabric config forwards on."""
    rows, cols = tuple(mesh_device.shape)
    node = lambda r, c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    hops = []
    if cluster_axis == 0:
        hops = [((r, c), (r + 1, c)) for c in range(cols) for r in range(rows - 1)]
    else:
        hops = [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    counts = []
    for a, b in hops:
        counts.append(len(ttnn.get_forwarding_link_indices(node(*a), node(*b))))
        counts.append(len(ttnn.get_forwarding_link_indices(node(*b), node(*a))))
    return min(counts) if counts else 0


def _skip_if_infeasible(mesh_device, a_shape, w_shape, cluster_axis, scatter_dim):
    g = _group_size(mesh_device, cluster_axis)
    if g < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has {g} device(s); nothing to reduce")
    extent = a_shape[-2] if scatter_dim == -2 else w_shape[-1]
    if extent % (32 * g):
        pytest.skip(f"scattered extent {extent} does not split into {g} tile-aligned blocks")
    return g


def _stacked_randn(mesh_device, shape, seed, scale=1.0):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    return (torch.randn((rows, cols, *shape), dtype=torch.float32) * scale).to(torch.bfloat16)


def _as_device_holds(stacked, dtype):
    """The tensor as the device will hold it (block-float rounding for bfloat8_b)."""
    if dtype == ttnn.bfloat8_b:
        t = ttnn.from_torch(stacked.reshape(-1, stacked.shape[-1]).float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)
    return stacked


def _to_mesh(stacked, mesh_device, dtype):
    """Device (r, c) receives stacked[r, c]."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _reference(a_stacked, w_stacked, cluster_axis, scatter_dim):
    """fp32 torch: per-device A @ W, summed over the group, device p keeps block p of the scattered dim."""
    a = a_stacked.float()
    w = w_stacked.float().reshape(*w_stacked.shape[:2], *([1] * (a.dim() - 4)), *w_stacked.shape[2:])
    partial = torch.matmul(a, w)
    total = partial.sum(dim=cluster_axis, keepdim=True).expand_as(partial)
    rows, cols = a.shape[0], a.shape[1]
    g = (rows, cols)[cluster_axis]
    blocks = [
        [torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[cluster_axis]] for c in range(cols)] for r in range(rows)
    ]
    return torch.stack([torch.stack(row) for row in blocks])


def _expected_shape(a_shape, w_shape, scatter_dim, g):
    out = [*a_shape[:-1], w_shape[-1]]
    out[scatter_dim] //= g
    return out


def _per_device(out, cols):
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        yield divmod(idx, cols), ttnn.to_torch(t).float()


def _pcc(x, y):
    x, y = x.flatten().double(), y.flatten().double()
    x, y = x - x.mean(), y - y.mean()
    den = x.norm() * y.norm()
    return 1.0 if den == 0 else float((x * y).sum() / den)


def _run_and_check(mesh_device, a_shape, w_shape, *, cluster_axis, scatter_dim, weight_dtype, seed=0, **kwargs):
    g = _skip_if_infeasible(mesh_device, a_shape, w_shape, cluster_axis, scatter_dim)
    a = _stacked_randn(mesh_device, a_shape, seed)
    w = _as_device_holds(_stacked_randn(mesh_device, w_shape, seed + 1, scale=w_shape[0] ** -0.5), weight_dtype)
    expected = _reference(a, w, cluster_axis, scatter_dim)

    out = matmul_reduce_scatter(
        _to_mesh(a, mesh_device, ttnn.bfloat16),
        _to_mesh(w, mesh_device, weight_dtype),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        **kwargs,
    )
    shape = _expected_shape(a_shape, w_shape, scatter_dim, g)
    assert list(out.shape) == shape, f"shape {list(out.shape)} != {shape}"
    assert out.dtype == ttnn.bfloat16, f"output dtype {out.dtype} != bfloat16"
    assert out.layout == ttnn.TILE_LAYOUT
    cols = tuple(mesh_device.shape)[1]
    for (r, c), actual in _per_device(out, cols):
        ref = expected[r, c]
        assert list(actual.shape) == list(ref.shape)
        assert torch.isfinite(actual).all(), f"device ({r},{c}): non-finite output"
        pcc = _pcc(actual, ref)
        assert pcc >= PCC[weight_dtype], f"device ({r},{c}): pcc {pcc:.6f} < {PCC[weight_dtype]}"


# --------------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------------

# Per-device (A, W) shapes. Every scattered extent splits into tile blocks for G <= 8 where possible.
SHAPES = [
    pytest.param((1, 1, 256, 256), (256, 256), id="single_tile_blocks"),
    pytest.param((1, 1, 512, 1024), (1024, 512), id="multi_tile"),
    pytest.param((1, 1, 256, 512), (512, 2048), id="non_square"),
    pytest.param((256, 384), (384, 512), id="rank2"),
    pytest.param((1, 512, 768), (768, 1024), id="rank3"),
]


@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["w_bf16", "w_bf8b"])
@pytest.mark.parametrize("scatter_dim", [-2, -1], ids=["rows", "cols"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
@pytest.mark.parametrize("a_shape,w_shape", SHAPES)
def test_matmul_reduce_scatter(mesh_device, a_shape, w_shape, cluster_axis, scatter_dim, weight_dtype):
    _run_and_check(
        mesh_device,
        a_shape,
        w_shape,
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        weight_dtype=weight_dtype,
        topology=ttnn.Topology.Linear,
        num_links=1,
    )


@pytest.mark.parametrize("scatter_dim", [-2, -1], ids=["rows", "cols"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_two_links(mesh_device, cluster_axis, scatter_dim):
    if _usable_links(mesh_device, cluster_axis) < 2:
        pytest.skip("fewer than 2 usable links per hop along this axis")
    _run_and_check(
        mesh_device,
        (1, 1, 512, 1024),
        (1024, 2048),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        weight_dtype=ttnn.bfloat8_b,
        topology=ttnn.Topology.Linear,
        num_links=2,
    )


def test_perf_focus_shape(mesh_device):
    """Kimi K2.7 o_proj geometry (hidden-dim scatter), production weight dtype, default links."""
    _run_and_check(
        mesh_device,
        (1, 1, 640, 2048),
        (2048, 7168),
        cluster_axis=1,
        scatter_dim=-1,
        weight_dtype=ttnn.bfloat8_b,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False
        ),
    )


def test_default_call(mesh_device):
    """matmul_reduce_scatter(a, w, cluster_axis=1): scatter_dim=-2, Linear, all usable links, default config."""
    _run_and_check(
        mesh_device, (1, 1, 512, 512), (512, 1024), cluster_axis=1, scatter_dim=-2, weight_dtype=ttnn.bfloat16
    )


def test_maxed_precision(mesh_device):
    _run_and_check(
        mesh_device,
        (1, 1, 256, 1024),
        (1024, 1024),
        cluster_axis=1,
        scatter_dim=-1,
        weight_dtype=ttnn.bfloat16,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )


@pytest.mark.parametrize("scatter_dim", [-2, -1], ids=["rows", "cols"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis0", "axis1"])
def test_every_partial_exactly_once(mesh_device, cluster_axis, scatter_dim):
    """A is ones on 32 K columns, W[r, c] is the constant rank r*C + c + 1 on those rows: every partial element is
    exactly 32 * rank and the group sum is an exact small integer in bf16. A dropped, doubled or foreign partial,
    or a block on the wrong device, is a wrong integer."""
    a_shape, w_shape = (1, 1, 512, 256), (256, 512)
    g = _skip_if_infeasible(mesh_device, a_shape, w_shape, cluster_axis, scatter_dim)
    rows, cols = tuple(mesh_device.shape)
    a = torch.zeros((rows, cols, *a_shape), dtype=torch.bfloat16)
    a[..., :32] = 1
    w = torch.zeros((rows, cols, *w_shape), dtype=torch.float32)
    w[:, :, :32, :] = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols, 1, 1)
    w = w.to(torch.bfloat16)
    expected = _reference(a, w, cluster_axis, scatter_dim)
    out = matmul_reduce_scatter(
        _to_mesh(a, mesh_device, ttnn.bfloat16),
        _to_mesh(w, mesh_device, ttnn.bfloat16),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=ttnn.Topology.Linear,
        num_links=1,
    )
    assert list(out.shape) == _expected_shape(a_shape, w_shape, scatter_dim, g)
    for (r, c), actual in _per_device(out, cols):
        bad = int((actual != expected[r, c].float()).sum())
        assert bad == 0, f"device ({r},{c}): {bad}/{actual.numel()} elements differ from the exact group sum"


def test_deterministic_and_reusable(mesh_device):
    """Back-to-back calls (no host sync between them) give bit-identical output on every device."""
    a_shape, w_shape = (1, 1, 512, 1024), (1024, 1024)
    _skip_if_infeasible(mesh_device, a_shape, w_shape, 1, -1)
    a = _to_mesh(_stacked_randn(mesh_device, a_shape, 7), mesh_device, ttnn.bfloat16)
    w = _to_mesh(_stacked_randn(mesh_device, w_shape, 8, scale=w_shape[0] ** -0.5), mesh_device, ttnn.bfloat16)
    outs = [
        matmul_reduce_scatter(a, w, cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Linear, num_links=1)
        for _ in range(4)
    ]
    cols = tuple(mesh_device.shape)[1]
    first = dict(_per_device(outs[0], cols))
    for out in outs[1:]:
        for rc, actual in _per_device(out, cols):
            assert torch.equal(actual, first[rc]), f"device {rc}: output differs between identical calls"


def test_num_links_above_discovered_is_value_error(mesh_device, expect_error):
    a_shape, w_shape = (1, 1, 256, 256), (256, 256)
    _skip_if_infeasible(mesh_device, a_shape, w_shape, 1, -2)
    usable = _usable_links(mesh_device, 1)
    if usable + 1 > 2:
        pytest.skip(
            f"{usable} usable links: the next count is outside SUPPORTED num_links (a support refusal, not this check)"
        )
    a = _to_mesh(_stacked_randn(mesh_device, a_shape, 0), mesh_device, ttnn.bfloat16)
    w = _to_mesh(_stacked_randn(mesh_device, w_shape, 1), mesh_device, ttnn.bfloat16)
    with expect_error(ValueError, ""):  # any message: a caller error, not a support refusal
        matmul_reduce_scatter(a, w, cluster_axis=1, num_links=usable + 1)


def test_k_mismatch_is_value_error(mesh_device, expect_error):
    a = _to_mesh(_stacked_randn(mesh_device, (1, 1, 256, 256), 0), mesh_device, ttnn.bfloat16)
    w = _to_mesh(_stacked_randn(mesh_device, (512, 256), 1), mesh_device, ttnn.bfloat16)
    with expect_error(ValueError, ""):  # any message: a caller error, not a support refusal
        matmul_reduce_scatter(a, w, cluster_axis=1)
