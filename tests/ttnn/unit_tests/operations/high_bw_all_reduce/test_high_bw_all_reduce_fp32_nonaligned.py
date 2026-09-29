# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — float32 end-to-end + non-tile-aligned shapes.

fp32: every CB page and the wire are Float32 and the add runs on the SFPU in fp32 DEST
(UnpackToDestFp32 operands). The exactness test uses values k * 2^-12 with k in [2049, 4095]
(12 significant bits): the group sum is exact in fp32 but NOT in tf32 (10 mantissa bits), so an
FPU add (SrcA/SrcB tf32) or a bf16 wire downcast would fail it.

Non-aligned: the op sums physical tile pages and the output reuses the input TensorSpec, so the
padded region never reaches the logical view. Rank-identity exactness pins it on every path.
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
        pytest.skip("needs a 2-D mesh")
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


TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# (cluster_axis, topology) cells enabled by Phase 0 + Refinement 1 on a 2-D mesh.
PATHS = [
    pytest.param(0, Topology.Linear, id="axis0-line"),
    pytest.param(1, Topology.Linear, id="axis1-line"),
    pytest.param(None, Topology.Linear, id="none-snake"),
    pytest.param(None, Topology.Ring, id="none-ring"),
]


def _to_mesh(stacked, mesh_device, dtype):
    rows, cols = stacked.shape[0], stacked.shape[1]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
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


def _run(mesh_device, stacked, cluster_axis, topology, num_links, dtype):
    out = high_bw_all_reduce(
        _to_mesh(stacked, mesh_device, dtype), cluster_axis=cluster_axis, topology=topology, num_links=num_links
    )
    assert list(out.shape) == list(stacked.shape[2:])
    assert out.dtype == dtype
    return out


def _per_device(out, cols):
    for idx, dev_tensor in enumerate(ttnn.get_device_tensors(out)):
        yield divmod(idx, cols), ttnn.to_torch(dev_tensor)


def _assert_exact(out, expected):
    for (r, c), actual in _per_device(out, expected.shape[1]):
        assert tuple(actual.shape) == tuple(expected.shape[2:])
        bad = (actual.float() != expected[r, c].float()).sum().item()
        assert bad == 0, f"device ({r},{c}): {bad} elements differ from the exact group sum"


@pytest.mark.parametrize("cluster_axis,topology", PATHS)
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 32), (3, 96, 160), (1, 1, 1024, 2048)], ids=lambda s: "x".join(map(str, s))
)
def test_fp32_beyond_tf32_exact(mesh_device, shape, cluster_axis, topology, num_links):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(5)
    k = torch.randint(2049, 4096, (rows, cols, *shape), dtype=torch.int32)
    stacked = k.to(torch.float32) * 2.0**-12
    out = _run(mesh_device, stacked, cluster_axis, topology, num_links, ttnn.float32)
    _assert_exact(out, _ref(stacked, cluster_axis))


@pytest.mark.parametrize("cluster_axis,topology", PATHS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "shape",
    [(4096, 2050), (4001, 2048), (1, 1, 48, 80), (2, 33, 65)],
    ids=["4096x2050", "4001x2048", "48x80", "2x33x65"],
)
def test_non_aligned_rank_identity(mesh_device, shape, dtype, cluster_axis, topology):
    """Device (r, c) holds r*C + c + 1 everywhere; the group sum is an exact small integer."""
    rows, cols = tuple(mesh_device.shape)
    ranks = torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols)
    view = (rows, cols) + (1,) * len(shape)
    stacked = ranks.reshape(view).expand(rows, cols, *shape).to(TORCH_DTYPE[dtype]).contiguous()
    out = _run(mesh_device, stacked, cluster_axis, topology, None, dtype)
    _assert_exact(out, _ref(stacked, cluster_axis))


@pytest.mark.parametrize("cluster_axis,topology", PATHS)
@pytest.mark.parametrize("shape", [(1, 1, 2048, 2048), (1, 1, 100, 1000)], ids=["2048x2048", "100x1000"])
def test_fp32_random(mesh_device, shape, cluster_axis, topology):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(9)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.float32) * 1e3
    expected = _ref(stacked, cluster_axis)
    out = _run(mesh_device, stacked, cluster_axis, topology, None, ttnn.float32)
    for (r, c), actual in _per_device(out, cols):
        want = expected[r, c]
        rel = ((actual.float() - want).norm() / want.norm()).item()
        assert rel < 1e-6, f"device ({r},{c}): rel_rms={rel:.3e} (fp32 end-to-end expected)"
        assert _pcc(actual, want) >= 0.99999
