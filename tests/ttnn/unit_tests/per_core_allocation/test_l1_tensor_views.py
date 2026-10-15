# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

import ttnn


def _spec(cores, width, per_core=True):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        memory.experimental_set_per_core_allocation(True)
    return ttnn.TensorSpec([len(cores), width], ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, memory)


def _owner(mesh, core, width=768, per_core=True):
    spec = _spec([core], width, per_core)
    return ttnn.from_torch(
        torch.zeros(1, width, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=spec.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _read(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


@pytest.mark.parametrize("per_core", [True, False])
def test_view_retains_owner_and_preserves_neighbor_bytes(per_core_mesh_device, per_core):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    baseline = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    owner = _owner(mesh, core, per_core=per_core)
    spec = _spec([core], 256)
    before = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    views = [
        ttnn.experimental_create_l1_tensor_view(
            mesh, spec, [owner], [ttnn.L1TensorViewPlacement(coord, core, 0, offset)]
        )
        for offset in (0, 256, 512)
    ]
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == before
    del owner
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == before
    values = [torch.full((1, 256), value, dtype=torch.uint8) for value in (17, 23, 29)]
    for view, value in zip(views, values):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=ttnn.uint8), view)
    for view, value in zip(views, values):
        torch.testing.assert_close(_read(view), value)
    del view, views
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline


@pytest.mark.parametrize("bad", ["owner", "core", "alignment", "bounds", "duplicate", "missing"])
def test_view_rejects_invalid_placements(per_core_mesh_device, bad):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    placement = ttnn.L1TensorViewPlacement(coord, core, 0, 0)
    if bad == "owner":
        placement = ttnn.L1TensorViewPlacement(coord, core, 1, 0)
    elif bad == "core":
        placement = ttnn.L1TensorViewPlacement(coord, ttnn.CoreCoord(1, 0), 0, 0)
    elif bad == "alignment":
        placement = ttnn.L1TensorViewPlacement(coord, core, 0, 1)
    elif bad == "bounds":
        (extent,) = ttnn.experimental_l1_tensor_owned_extents(owner)
        placement = ttnn.L1TensorViewPlacement(coord, core, 0, extent.size)
    placements = [] if bad == "missing" else [placement]
    if bad == "duplicate":
        placements.append(placement)
    with pytest.raises(RuntimeError):
        ttnn.experimental_create_l1_tensor_view(mesh, spec, [owner], placements)


def test_nested_view_cannot_borrow_parent_capacity(per_core_mesh_device):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    view = ttnn.experimental_create_l1_tensor_view(
        mesh, spec, [owner], [ttnn.L1TensorViewPlacement(coord, core, 0, 256)]
    )
    (extent,) = ttnn.experimental_l1_tensor_owned_extents(view)
    assert extent.size == ttnn.experimental_l1_tensor_geometry(mesh, coord, spec).aligned_shard_size
    with pytest.raises(RuntimeError):
        ttnn.experimental_create_l1_tensor_view(
            mesh, spec, [view], [ttnn.L1TensorViewPlacement(coord, core, 0, 256)]
        )


def test_lockstep_view_requires_one_address(per_core_mesh_device):
    mesh = per_core_mesh_device
    coord = ttnn.MeshCoordinate(0, 0)
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    owner = _owner(mesh, cores[0], per_core=False)
    spec = _spec(cores, 256, per_core=False)
    view = ttnn.experimental_create_l1_tensor_view(
        mesh, spec, [owner], [ttnn.L1TensorViewPlacement(coord, core, 0, 0) for core in cores]
    )
    assert view.buffer_address() == owner.buffer_address()
    with pytest.raises(RuntimeError):
        ttnn.experimental_create_l1_tensor_view(
            mesh,
            spec,
            [owner],
            [ttnn.L1TensorViewPlacement(coord, core, 0, i * 256) for i, core in enumerate(cores)],
        )


def test_free_ranges_exclude_live_owner(per_core_mesh_device):
    mesh = per_core_mesh_device
    core = ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    (extent,) = ttnn.experimental_l1_tensor_owned_extents(owner)
    free = ttnn.experimental_get_l1_free_ranges(mesh, extent.device_coord, core)
    assert free and all(start < end for start, end in free)
    assert all(end <= extent.address or start >= extent.address + extent.size for start, end in free)


def test_view_preserves_selected_device_coverage(mesh_device):
    mesh = mesh_device
    coord, core = ttnn.MeshCoordinate(0, 1), ttnn.CoreCoord(0, 0)
    spec = _spec([core], 256)
    host = ttnn.from_torch(torch.arange(256, dtype=torch.uint8).reshape(1, 256), dtype=ttnn.uint8)
    owner = ttnn.experimental_to_single_device(host, mesh, coord, spec.memory_config)
    view = ttnn.experimental_create_l1_tensor_view(
        mesh, spec, [owner], [ttnn.L1TensorViewPlacement(coord, core, 0, 0)]
    )
    assert view.device_coords() == [coord]
    assert all(extent.device_coord == coord for extent in ttnn.experimental_l1_tensor_owned_extents(view))
    del owner
    gc.collect()
    torch.testing.assert_close(_read(view), torch.arange(256, dtype=torch.uint8).reshape(1, 256))
