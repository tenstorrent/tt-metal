# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn

from ...parallel.manager import CCLManager

LINE_AXIS_1 = {1: ttnn.Topology.Linear}


def _manager(mesh_shape: tuple[int, int], topology: ttnn.Topology, axis_topologies=None) -> CCLManager:
    manager = CCLManager.__new__(CCLManager)  # no device: axis_topology reads only the mesh shape and the topologies
    manager.mesh_device = SimpleNamespace(shape=mesh_shape)
    manager.topology = topology
    manager.axis_topologies = axis_topologies or {}
    return manager


def _ccl_manager(mesh_shape: tuple[int, int], monkeypatch, axis_topologies=None) -> tuple[CCLManager, dict]:
    """A Ring-mesh manager whose all-gather / reduce-scatter ops only record the topology they are given."""
    manager = _manager(mesh_shape, ttnn.Topology.Ring, axis_topologies)
    manager.num_links = 1
    for name in ("get_barrier_semaphore", "get_ag_ping_pong_semaphore", "get_rs_ping_pong_semaphore"):
        setattr(manager, name, lambda mesh_axis: None)
    calls = {}

    def record(op):
        def run(tensor, **kwargs):
            calls[op] = kwargs["topology"]
            return tensor

        return run

    monkeypatch.setattr(ttnn.experimental, "all_gather_async", record("all_gather"))
    monkeypatch.setattr(ttnn.experimental, "reduce_scatter_minimal_async", record("reduce_scatter"))
    return manager, calls


@pytest.mark.parametrize(
    ("mesh_shape", "topology", "axis_topologies", "mesh_axis", "expected"),
    [
        ((4, 8), ttnn.Topology.Ring, None, 0, ttnn.Topology.Ring),
        ((4, 8), ttnn.Topology.Ring, None, 1, ttnn.Topology.Ring),
        ((2, 8), ttnn.Topology.Ring, None, 0, ttnn.Topology.Linear),
        ((2, 8), ttnn.Topology.Ring, None, 1, ttnn.Topology.Ring),
        ((4, 2), ttnn.Topology.Ring, None, 0, ttnn.Topology.Ring),
        ((4, 2), ttnn.Topology.Ring, None, 1, ttnn.Topology.Linear),
        ((4, 2), ttnn.Topology.Linear, None, 0, ttnn.Topology.Linear),
        ((4, 2), ttnn.Topology.Ring, None, None, ttnn.Topology.Ring),
        ((4, 4), ttnn.Topology.Ring, None, 1, ttnn.Topology.Ring),
        ((4, 4), ttnn.Topology.Ring, LINE_AXIS_1, 0, ttnn.Topology.Ring),
        ((4, 4), ttnn.Topology.Ring, LINE_AXIS_1, 1, ttnn.Topology.Linear),
    ],
)
def test_axis_topology(mesh_shape, topology, axis_topologies, mesh_axis, expected) -> None:
    assert _manager(mesh_shape, topology, axis_topologies).axis_topology(mesh_axis) == expected


@pytest.mark.parametrize(
    ("mesh_shape", "axis_topologies", "mesh_axis", "expected"),
    [
        ((4, 2), None, 0, ttnn.Topology.Ring),
        ((4, 2), None, 1, ttnn.Topology.Linear),
        ((4, 4), LINE_AXIS_1, 0, ttnn.Topology.Ring),
        ((4, 4), LINE_AXIS_1, 1, ttnn.Topology.Linear),
    ],
)
def test_all_gather_and_reduce_scatter_take_the_axis_topology(
    monkeypatch, mesh_shape, axis_topologies, mesh_axis, expected
) -> None:
    manager, calls = _ccl_manager(mesh_shape, monkeypatch, axis_topologies)
    x = SimpleNamespace(shape=[1, 1, 64, 64])
    manager.all_gather(x, dim=3, mesh_axis=mesh_axis, use_hyperparams=False)
    manager.reduce_scatter(x, dim=3, mesh_axis=mesh_axis)
    assert calls == {"all_gather": expected, "reduce_scatter": expected}


@pytest.mark.parametrize(
    ("mesh_shape", "axis_topologies", "mesh_axis", "intermediate_rows"),
    [((4, 2), None, 0, 1), ((4, 2), None, 1, 2), ((4, 4), LINE_AXIS_1, 0, 1), ((4, 4), LINE_AXIS_1, 1, 2)],
)
def test_reduce_scatter_intermediate_holds_two_slices_on_a_line_axis(
    monkeypatch, mesh_shape, axis_topologies, mesh_axis, intermediate_rows
) -> None:
    manager = _manager(mesh_shape, ttnn.Topology.Ring, axis_topologies)
    manager._ping_pong_buffer_cache, manager._ping_pong_buffer_indices = {}, {}
    shapes = []
    monkeypatch.setattr(ttnn, "allocate_tensor_on_device", lambda shape, *args: shapes.append(list(shape)))
    manager.get_rs_ping_pong_buffer([1, 1, 64, 128], 3, mesh_axis, device_synchronize=False)
    intermediate = [1, 1, 64, 128]
    intermediate[0] *= intermediate_rows
    assert shapes[0] == intermediate
