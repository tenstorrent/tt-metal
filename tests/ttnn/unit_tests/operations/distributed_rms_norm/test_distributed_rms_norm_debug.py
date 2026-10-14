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
