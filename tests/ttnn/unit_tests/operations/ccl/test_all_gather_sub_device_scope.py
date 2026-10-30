# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather records the sub-device it runs on: its subdevice_id, or the sub-device scope's."""

import re
from contextlib import contextmanager, nullcontext

import pytest
import torch
import ttnn


def core_range(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


@contextmanager
def compute_and_last_column(mesh_device):
    """Sub-device 0 for compute, 1 (the last column) for collectives."""
    grid = mesh_device.compute_with_storage_grid_size()
    compute = core_range(0, 0, grid.x - 2, grid.y - 1)
    last_column = core_range(grid.x - 1, 0, grid.x - 1, grid.y - 1)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([compute]), ttnn.SubDevice([last_column])], 0)
    mesh_device.load_sub_device_manager(manager)
    try:
        yield compute, last_column
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)


def program_placements(fn):
    ttnn.graph.begin_graph_capture()
    try:
        result = fn()
    finally:
        graph = ttnn.graph.end_graph_capture()
    return result, {node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"}


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "sub_device_id, by_scope",
    [(1, True), (0, True), (1, False)],
    ids=["scope_last_column", "scope_compute", "subdevice_id_argument"],
)
def test_all_gather_runs_on_its_lane(mesh_device, sub_device_id, by_scope):
    torch_input = torch.randn(1, 1, 64, 256, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    kwargs = {} if by_scope else {"subdevice_id": ttnn.SubDeviceId(sub_device_id)}
    with compute_and_last_column(mesh_device):
        with ttnn.sub_device(sub_device_id) if by_scope else nullcontext():
            y, sub_devices = program_placements(
                lambda: ttnn.all_gather(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear, **kwargs)
            )
        ttnn.synchronize_device(mesh_device)
        assert sub_devices == {sub_device_id}
        for gathered in ttnn.get_device_tensors(y):
            assert torch.equal(ttnn.to_torch(gathered), torch_input)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_gather_with_another_sub_device_than_the_scope_raises(mesh_device, expect_error):
    x = ttnn.from_torch(
        torch.randn(1, 1, 64, 256, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    with compute_and_last_column(mesh_device):
        with ttnn.sub_device(1):
            with expect_error(
                RuntimeError, "was asked to run on sub-device 0 inside a sub-device scope on sub-device 1"
            ):
                ttnn.all_gather(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear, subdevice_id=ttnn.SubDeviceId(0))


def sharded_on_mesh(mesh_device, torch_input):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_gather_refuses_a_sub_core_grid_outside_its_sub_device(mesh_device, expect_error):
    x = sharded_on_mesh(mesh_device, torch.randn(1, 1, 64, 256, dtype=torch.bfloat16))
    with compute_and_last_column(mesh_device) as (compute, _):
        with expect_error(RuntimeError, "outside sub-device 1"):
            ttnn.all_gather(
                x,
                3,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                subdevice_id=ttnn.SubDeviceId(1),
                sub_core_grids=compute,
            )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_the_composite_all_gather_carries_its_sub_device_to_the_ops_that_record_one(mesh_device, expect_error):
    """A gather dimension that is not tile-aligned takes the composite path: typecast, all_broadcast, concat, typecast.
    Given subdevice_id, the typecasts follow the gather's sub-device, all_broadcast takes subdevice_id itself, and
    concat, which records no sub-device, runs where its own code puts it. Inside a scope the call is refused at
    all_broadcast, which records none either, after the first typecast ran on the scope's sub-device; the refusal
    names all_gather, the call that launched it."""
    torch_input = torch.randn(1, 1, 32, 96, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    with compute_and_last_column(mesh_device):
        ttnn.graph.begin_graph_capture()
        try:
            ttnn.all_gather(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear, subdevice_id=ttnn.SubDeviceId(1))
        finally:
            graph = ttnn.graph.end_graph_capture()
        ttnn.synchronize_device(mesh_device)
        # One program per device each: typecast, all_broadcast, concat, typecast.
        placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
        assert len(placements) == 8, placements
        assert placements[:4] == [1, 1, 1, 1] and placements[6:] == [1, 1], placements
        with ttnn.sub_device(1):
            ttnn.graph.begin_graph_capture()
            try:
                with expect_error(
                    RuntimeError,
                    re.escape(", launched within all_gather, cannot run inside a sub-device scope (sub-device 1)"),
                ) as refused:
                    ttnn.all_gather(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear)
            finally:
                graph = ttnn.graph.end_graph_capture()
        assert "all_gather cannot run inside a sub-device scope on this path" in str(refused.value)
        # What ran before the refusal is not undone: the first typecast, one program per device.
        placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
        assert placements == [1, 1], placements


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_the_composite_all_gather_checks_its_sub_device_against_the_scope_before_launching_anything(
    mesh_device, expect_error
):
    """Its subdevice_id contradicts the scope, so the call is refused at its entry, before the first typecast runs."""
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 96, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    with compute_and_last_column(mesh_device):
        with ttnn.sub_device(1):
            ttnn.graph.begin_graph_capture()
            try:
                with expect_error(
                    RuntimeError,
                    "all_gather was asked to run on sub-device 0 inside a sub-device scope on sub-device 1",
                ):
                    ttnn.all_gather(
                        x, 3, cluster_axis=1, topology=ttnn.Topology.Linear, subdevice_id=ttnn.SubDeviceId(0)
                    )
            finally:
                graph = ttnn.graph.end_graph_capture()
        assert not [node for node in graph if node["node_type"] == "program_execution"]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_the_composite_all_gather_runs_on_the_sub_device_its_sub_core_grids_name(mesh_device):
    """With no subdevice_id and no scope, sub_core_grids names the gather's sub-device on the composite path too: the
    typecasts follow it, and so does all_broadcast, which takes the id the guard resolved. concat records none."""
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 96, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    with compute_and_last_column(mesh_device) as (_, last_column):
        ttnn.graph.begin_graph_capture()
        try:
            ttnn.all_gather(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear, sub_core_grids=last_column)
        finally:
            graph = ttnn.graph.end_graph_capture()
        ttnn.synchronize_device(mesh_device)
    # One program per device each: typecast, all_broadcast, concat, typecast.
    placements = [node["params"].get("sub_device_id") for node in graph if node["node_type"] == "program_execution"]
    assert len(placements) == 8, placements
    assert placements[:4] == [1, 1, 1, 1] and placements[6:] == [1, 1], placements


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_a_composite_collective_named_by_no_argument_is_named_inside_a_scope(mesh_device, expect_error):
    """reduce_scatter to a width that is not tile-aligned takes the composite path, whose helper builds its guard inside
    a scope even without a subdevice_id. Its first operation that records no sub-device is refused naming it."""
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 192, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    with compute_and_last_column(mesh_device):
        with ttnn.sub_device(1):
            with expect_error(
                RuntimeError,
                re.escape(
                    ", launched within composite_reduce_scatter, cannot run inside a sub-device scope (sub-device 1)"
                ),
            ) as refused:
                ttnn.reduce_scatter(x, 3, cluster_axis=1, topology=ttnn.Topology.Linear)
    assert "Launch it outside the scope" not in str(refused.value)
