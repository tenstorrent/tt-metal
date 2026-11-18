# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn

from ...layers.linear import RowParallelLinear
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


def test_fused_addcmul_on_a_2_chip_axis_runs_matmul_reduce_scatter_then_addcmul(monkeypatch) -> None:
    linear = RowParallelLinear.__new__(RowParallelLinear)  # no device: the line-axis branch only routes the call
    linear.ccl_manager = _manager((4, 2), ttnn.Topology.Ring)
    linear.mesh_axis = 1
    calls = []
    linear.forward = lambda x, **kwargs: calls.append(("forward", x, kwargs["dtype"])) or "rs"

    def fused(**kwargs):
        raise AssertionError("the Ring-only fused op ran on a line axis")

    monkeypatch.setattr(ttnn.experimental, "minimal_matmul_strided_reduce_scatter_async", fused)
    monkeypatch.setattr(ttnn, "addcmul", lambda a, t1, t2, value: calls.append(("addcmul", a, t1, t2, value)) or "out")
    assert linear.forward_fused_addcmul(["prefix", "suffix"], "a", "b", 0.5, dtype="bf16") == "out"
    assert calls == [("forward", ["prefix", "suffix"], "bf16"), ("addcmul", "a", "rs", "b", 0.5)]
