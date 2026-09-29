# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — whole-mesh snake line (R2) and rotated-chain ring (R3) over FABRIC_2D.

cluster_axis=None makes the whole mesh one group of G = rows*cols devices (G >= 3 exercises the
`middle` role). Linear runs the row-snake line; Ring runs the snake Hamiltonian cycle with one
rotated chain per ring position. Exactness tests use small-integer data (exact in bf16).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.ccl import Topology
from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _system_mesh_shape()
    if rows < 2 or cols < 2:
        pytest.skip("cluster_axis=None snake needs a 2-D mesh")
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


def _to_mesh(stacked, mesh_device):
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _ref(stacked, cluster_axis):
    x = stacked.to(torch.float32)
    dims = (0, 1) if cluster_axis is None else (cluster_axis,)
    return x.sum(dim=dims, keepdim=True).expand_as(x).to(stacked.dtype)


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a * b).sum() / den) if den > 1e-30 else 1.0


def _per_device(out, cols):
    for idx, dev_tensor in enumerate(ttnn.get_device_tensors(out)):
        yield divmod(idx, cols), ttnn.to_torch(dev_tensor).float()


def _run(mesh_device, stacked, cluster_axis, topology, num_links):
    out = high_bw_all_reduce(
        _to_mesh(stacked, mesh_device), cluster_axis=cluster_axis, topology=topology, num_links=num_links
    )
    assert list(out.shape) == list(stacked.shape[2:])
    return out


def _assert_exact(out, expected):
    for (r, c), actual in _per_device(out, expected.shape[1]):
        bad = (actual != expected[r, c].float()).sum().item()
        assert bad == 0, f"device ({r},{c}): {bad} elements differ from the exact group sum"


TOPOLOGIES = [Topology.Linear, Topology.Ring]
TOPO_IDS = ["linear", "ring"]

SHAPES = [
    (1, 1, 32, 32),  # single tile: most slices / reducers idle
    (3, 96, 160),  # 45 tiles, ragged last chunk
    (1, 1, 256, 512),  # 128 tiles
    (1, 1, 2048, 2048),  # 8 MB: many chunks per slice
]


@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPO_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_whole_mesh(mesh_device, shape, topology, num_links):
    torch.manual_seed(3)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    expected = _ref(stacked, None)
    out = _run(mesh_device, stacked, None, topology, num_links)
    for (r, c), actual in _per_device(out, cols):
        pcc = _pcc(actual, expected[r, c])
        assert pcc >= 0.995, f"device ({r},{c}): pcc={pcc:.6f}"


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPO_IDS)
@pytest.mark.parametrize("shape", [(1, 1, 256, 512), (1, 1, 2048, 4096)], ids=["256x512", "2048x4096"])
def test_rank_identity_exact(mesh_device, shape, topology):
    """Device (r, c) holds r*C + c + 1; the whole-mesh sum is an exact small integer."""
    rows, cols = tuple(mesh_device.shape)
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    stacked = ranks.reshape(rows, cols, 1, 1, 1, 1).expand(rows, cols, *shape).to(torch.bfloat16)
    _assert_exact(_run(mesh_device, stacked, None, topology, None), _ref(stacked, None))


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPO_IDS)
@pytest.mark.parametrize("contributor", ["first", "last", "middle"])
def test_single_contributor_exact(mesh_device, topology, contributor):
    """One device holds data, the rest zero: every device must receive exactly that data."""
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 512, 1024)
    torch.manual_seed(11)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    who = {"first": (0, 0), "last": (rows - 1, 0), "middle": (0, cols - 1)}[contributor]
    mask = torch.zeros((rows, cols), dtype=torch.bfloat16)
    mask[who] = 1
    stacked = stacked * mask.reshape(rows, cols, 1, 1, 1, 1)
    _assert_exact(_run(mesh_device, stacked, None, topology, 1), _ref(stacked, None))


def test_linear_ring_alternation(mesh_device):
    """Back-to-back invocations alternating Linear <-> Ring (and the axis lines) must not leak
    counter / credit state between configs."""
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 512, 1024)
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    stacked = ranks.reshape(rows, cols, 1, 1, 1, 1).expand(rows, cols, *shape).to(torch.bfloat16)
    seq = [
        (None, Topology.Ring),
        (None, Topology.Linear),
        (None, Topology.Ring),
        (0, Topology.Linear),
        (None, Topology.Ring),
        (1, Topology.Linear),
        (None, Topology.Linear),
        (None, Topology.Ring),
        (None, Topology.Ring),
    ]
    for cluster_axis, topology in seq:
        _assert_exact(_run(mesh_device, stacked, cluster_axis, topology, 1), _ref(stacked, cluster_axis))


def test_axis_ring_refused(mesh_device, expect_error):
    """Per-axis rings (torus wrap links) are in EXCLUSIONS: a support refusal, not a hang."""
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.zeros((rows, cols, 1, 1, 64, 64), dtype=torch.bfloat16)
    with expect_error(NotImplementedError, ""):
        high_bw_all_reduce(_to_mesh(stacked, mesh_device), cluster_axis=0, topology=Topology.Ring, num_links=1)
