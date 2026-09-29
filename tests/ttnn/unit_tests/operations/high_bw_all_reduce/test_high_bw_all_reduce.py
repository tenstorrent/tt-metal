# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for high_bw_all_reduce (Phase 0 contract).

IMMUTABLE SPEC — the implementer must not modify this file.

Phase 0 SUPPORTED: bfloat16, TILE, tile_aligned, cluster_axis in {0, 1},
topology Linear, num_links in {1, 2}. Every device of the system mesh holds
distinct data; every device must receive the SUM over its collective group
(cluster_axis=0 -> its mesh column, 1 -> its mesh row).

The mesh is opened ONCE per module with FABRIC_2D (the op never changes the
fabric config itself). Cells the live cluster cannot exercise (an axis of
size 1, fewer usable links than requested) are skipped, not failed.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


# --- mesh fixture -------------------------------------------------------------


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _system_mesh_shape()
    if rows * cols < 2:
        pytest.skip("high_bw_all_reduce needs a multi-device mesh")
    from tests.scripts.common import get_updated_device_params

    fabric_config = ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT)
    device_params = get_updated_device_params({"fabric_config": fabric_config})
    device_params.pop("fabric_config")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **device_params)
    try:
        yield mesh
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# --- helpers ------------------------------------------------------------------


def _groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _usable_links(mesh_device, cluster_axis):
    """min over every hop (both directions) of the forwarding link count."""
    best = None
    for group in _groups(tuple(mesh_device.shape), cluster_axis):
        for a, b in zip(group, group[1:]):
            na = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*a))
            nb = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*b))
            for s, d in ((na, nb), (nb, na)):
                n = len(ttnn.get_forwarding_link_indices(s, d))
                best = n if best is None else min(best, n)
    return best or 0


def _skip_if_infeasible(mesh_device, cluster_axis, num_links):
    size = tuple(mesh_device.shape)[cluster_axis]
    if size < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has size {size}; nothing to reduce")
    if num_links is not None and num_links > _usable_links(mesh_device, cluster_axis):
        pytest.skip(f"num_links={num_links} exceeds the usable links on this cluster")


def _to_mesh(stacked, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """stacked[r, c] -> device (r, c)."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def torch_all_reduce(stacked, cluster_axis):
    """Reference: fp32 group sum, broadcast back to every member, cast to input dtype."""
    x = stacked.to(torch.float32)
    return x.sum(dim=cluster_axis, keepdim=True).expand_as(x).to(stacked.dtype)


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    if den < 1e-30:
        return 1.0 if (a - b).abs().max() < 1e-30 else 0.0
    return float((a * b).sum() / den)


def _check(output, expected, shape, dtype):
    assert list(output.shape) == list(shape), f"shape {list(output.shape)} != {list(shape)}"
    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    cols = expected.shape[1]
    device_tensors = ttnn.get_device_tensors(output)
    assert len(device_tensors) == expected.shape[0] * cols
    for idx, dev_tensor in enumerate(device_tensors):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(dev_tensor)
        want = expected[r, c]
        assert torch.isfinite(actual.float()).all(), f"device ({r},{c}): non-finite output"
        pcc = _pcc(actual, want)
        assert pcc >= PCC[dtype], f"device ({r},{c}): pcc={pcc:.6f} < {PCC[dtype]}"


def _run(mesh_device, shape, cluster_axis, num_links, seed=42):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    expected = torch_all_reduce(stacked, cluster_axis)
    out = high_bw_all_reduce(
        _to_mesh(stacked, mesh_device),
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        num_links=num_links,
    )
    _check(out, expected, shape, ttnn.bfloat16)


# --- tests --------------------------------------------------------------------

SHAPES = [
    (1, 1, 32, 32),  # single tile: fewer chunks than lanes x reducers (degenerate split)
    (1, 1, 256, 512),  # multi-tile, 128 tiles: multiple chunks per lane
    (1, 1, 64, 2048),  # non-square, wide
    (2, 4, 128, 256),  # multi-batch rank 4 (leading dims fold into the tile axis)
    (512, 1024),  # rank 2
    (3, 96, 160),  # rank 3, 45 tiles: ragged last chunk per lane
    (1, 1, 2048, 2048),  # 8 MB bf16: the bandwidth regime
]


@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_high_bw_all_reduce(mesh_device, shape, cluster_axis, num_links):
    _skip_if_infeasible(mesh_device, cluster_axis, num_links)
    _run(mesh_device, shape, cluster_axis, num_links)


@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_default_num_links(mesh_device, cluster_axis):
    """num_links=None uses every usable link."""
    _skip_if_infeasible(mesh_device, cluster_axis, None)
    _run(mesh_device, (1, 1, 512, 1024), cluster_axis, None)


@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_rank_identity_exact(mesh_device, cluster_axis):
    """Device (r, c) holds the constant r*C + c + 1: the group sum is a small
    integer, exact in bf16, so a dropped / doubled / foreign contribution is
    an exact mismatch on the device that got it."""
    _skip_if_infeasible(mesh_device, cluster_axis, 1)
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 256, 512)
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    stacked = ranks.reshape(rows, cols, 1, 1, 1, 1).expand(rows, cols, *shape).to(torch.bfloat16)
    expected = torch_all_reduce(stacked, cluster_axis)
    out = high_bw_all_reduce(
        _to_mesh(stacked, mesh_device), cluster_axis=cluster_axis, topology=ttnn.Topology.Linear, num_links=1
    )
    for idx, dev_tensor in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(dev_tensor).float()
        bad = (actual != expected[r, c].float()).sum().item()
        assert bad == 0, f"device ({r},{c}): {bad} elements differ from the exact group sum"


def test_back_to_back_invocations(mesh_device):
    """Repeated calls (same and alternating configs) must not leak semaphore /
    credit state from one invocation into the next."""
    rows, cols = tuple(mesh_device.shape)
    axes = [a for a in (0, 1) if (rows, cols)[a] >= 2]
    if not axes:
        pytest.skip("no axis with >= 2 devices")
    shapes = [(1, 1, 512, 1024), (1, 1, 64, 64), (1, 1, 1024, 512)]
    seed = 0
    for axis in axes + axes:
        for shape in shapes:
            seed += 1
            _run(mesh_device, shape, axis, 1, seed=seed)


@pytest.mark.parametrize(
    "kwargs",
    [
        # float32 / w_non_aligned were refusals until Refinement 2 added them to SUPPORTED.
        {"dtype": ttnn.bfloat8_b},  # dtype outside SUPPORTED
        {"layout": ttnn.ROW_MAJOR_LAYOUT},  # layout outside SUPPORTED
        {"topology": ttnn.Topology.Ring},  # per-axis Ring is in EXCLUSIONS
    ],
    ids=["bfloat8_b", "row_major", "ring"],
)
def test_support_refusal(mesh_device, kwargs, expect_error):
    """Cells outside SUPPORTED raise a support refusal (NotImplementedError subclass)."""
    rows, cols = tuple(mesh_device.shape)
    axis = 0 if rows >= 2 else 1
    shape = kwargs.get("shape", (1, 1, 64, 64))
    dtype = kwargs.get("dtype", ttnn.bfloat16)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    stacked = torch.randn((rows, cols, *shape), dtype=torch_dtype)
    t = _to_mesh(stacked, mesh_device, dtype=dtype, layout=kwargs.get("layout", ttnn.TILE_LAYOUT))
    # Any message: the refusal is identified by its type (SupportRefusal subclasses NotImplementedError).
    with expect_error(NotImplementedError, ""):
        high_bw_all_reduce(t, cluster_axis=axis, topology=kwargs.get("topology", ttnn.Topology.Linear), num_links=1)
