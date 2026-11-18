# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn

from ...parallel.manager import CCLManager


def _manager(mesh_shape: tuple[int, int], topology: ttnn.Topology) -> CCLManager:
    manager = CCLManager.__new__(CCLManager)  # no device: axis_topology reads only the mesh shape and the topology
    manager.mesh_device = SimpleNamespace(shape=mesh_shape)
    manager.topology = topology
    return manager


@pytest.mark.parametrize(
    ("mesh_shape", "topology", "mesh_axis", "expected"),
    [
        ((4, 8), ttnn.Topology.Ring, 0, ttnn.Topology.Ring),
        ((4, 8), ttnn.Topology.Ring, 1, ttnn.Topology.Ring),
        ((4, 2), ttnn.Topology.Ring, 0, ttnn.Topology.Ring),
        ((4, 2), ttnn.Topology.Ring, 1, ttnn.Topology.Linear),
        ((4, 2), ttnn.Topology.Linear, 0, ttnn.Topology.Linear),
        ((4, 2), ttnn.Topology.Ring, None, ttnn.Topology.Ring),
    ],
)
def test_axis_topology(mesh_shape, topology, mesh_axis, expected) -> None:
    assert _manager(mesh_shape, topology).axis_topology(mesh_axis) == expected
