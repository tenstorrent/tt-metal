# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug tests for distributed_rms_norm. DO NOT DELETE — documents the bring-up/debugging process."""

import pytest
import torch
import ttnn


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
    return cfg


SYSTEM_MESH = _system_mesh_shape()
FABRIC_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": _router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
}

_MESH = pytest.mark.parametrize("mesh_device", [SYSTEM_MESH], indirect=True)
_FABRIC = pytest.mark.parametrize("device_params", [FABRIC_PARAMS], indirect=True)


@_MESH
@_FABRIC
@pytest.mark.parametrize("rep", [0, 1])
def test_info(mesh_device, rep):
    """Print the live mesh / grid / link facts the plan depends on."""
    shape = tuple(mesh_device.shape)
    grid = mesh_device.compute_with_storage_grid_size()
    print("\nMESH", shape, "GRID", grid.x, grid.y, "PAYLOAD", ttnn.get_tt_fabric_max_payload_size_bytes())
    print("FABRIC", ttnn.get_fabric_config())
    node = lambda r, c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    for a in [(r, c) for r in range(shape[0]) for c in range(shape[1])]:
        for b in [(r, c) for r in range(shape[0]) for c in range(shape[1])]:
            if a != b:
                print(a, "->", b, list(ttnn.get_forwarding_link_indices(node(*a), node(*b))))
    for x, y in [(0, 0), (12, 0), (0, 1), (12, 9)]:
        if x < grid.x and y < grid.y:
            v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            print("logical", (x, y), "virt", (v.x, v.y))


def _run_rows(mesh_device, shape, cluster_axis=1, dtype=ttnn.bfloat16):
    """All-ones x, no gamma: every output element must be rsqrt(1 + eps) ~= 1. Returns per-tile-row max |err|."""
    from ttnn.operations.distributed_rms_norm import distributed_rms_norm

    rows, cols = tuple(mesh_device.shape)
    x = torch.ones((rows, cols, *shape), dtype=torch.float32)
    # make every tile row distinguishable: row r scaled by (1 + (r // 32) % 7)
    scale = torch.tensor([1.0 + ((r // 32) % 7) for r in range(shape[-2])]).reshape(-1, 1)
    x = x * scale
    glob = torch.cat([torch.cat(list(x[r]), dim=1) for r in range(rows)], dim=0)
    tx = ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(-1, shape[-1])
    # expected: x / sqrt(mean(x^2)) = 1 everywhere (each row is constant)
    err = (got - 1.0).abs()
    per_tile = err.reshape(-1, 32, shape[-1] // 32, 32).amax(dim=(1, 3))  # (tile rows, w tiles)
    print("\nPER-TILE ERR (first 4 tile rows) > 0.05:")
    for r in range(min(8, per_tile.shape[0])):
        print(r, "".join("X" if v > 0.05 else "." for v in per_tile[r].tolist()))
    return per_tile.amax(dim=1)


@_MESH
@_FABRIC
def test_four_rows_per_core(mesh_device):
    """4096x1024: S_w = 3, 4 tile rows per worker -> combine REDUCE_COL spans two DEST chunks."""
    per_row = _run_rows(mesh_device, (1, 1, 4096, 1024))
    bad = [i for i, e in enumerate(per_row.tolist()) if e > 0.05]
    print("\nBAD TILE ROWS", len(bad), bad[:40])
    print("ERR", [round(v, 3) for v in per_row.tolist()[:16]])
    assert not bad
