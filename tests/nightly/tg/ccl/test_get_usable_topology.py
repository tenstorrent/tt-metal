# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_get_usable_topology_ring_available(mesh_device, device_params):
    """On a TG ring fabric, a tensor spanning the full 8-device wraparound (cluster_axis 0) must resolve to Ring."""
    tensor = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Defaults to the fabric topology (Ring), and an explicit Ring request is preserved on a wrapping placement.
    assert ttnn.get_usable_topology(tensor, cluster_axis=0) == ttnn.Topology.Ring
    assert ttnn.get_usable_topology(tensor, topology=ttnn.Topology.Ring, cluster_axis=0) == ttnn.Topology.Ring
