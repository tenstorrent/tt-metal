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


def _view(mesh, spec, owner, placements):
    """Create an address-based view after resolving this test's backing offsets."""
    extents = {(tuple(e.device_coord), (e.core_coord.x, e.core_coord.y)): e for e in ttnn.experimental_l1_tensor_owned_extents(owner)}
    addresses = [ttnn.L1TensorViewPlacement(coord, core, extents[tuple(coord), (core.x, core.y)].address + offset)
                 for coord, core, offset in placements]
    return ttnn.experimental_create_l1_tensor_view(mesh, spec, addresses)


@pytest.mark.parametrize("per_core", [True, False])
def test_view_retains_owner_and_preserves_neighbor_bytes(per_core_mesh_device, per_core):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    baseline = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    owner = _owner(mesh, core, per_core=per_core)
    spec = _spec([core], 256)
    before = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    views = [
        _view(mesh, spec, owner, [(coord, core, offset)])
        for offset in (0, 256, 512)
    ]
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == before
    for view in views:
        ttnn.experimental_retain_tensor_backing(view, [owner])
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


@pytest.mark.parametrize("bad", ["core", "alignment", "duplicate", "missing"])
def test_view_rejects_invalid_placements(per_core_mesh_device, bad):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    (extent,) = ttnn.experimental_l1_tensor_owned_extents(owner)
    placement = ttnn.L1TensorViewPlacement(coord, core, extent.address)
    if bad == "core":
        placement = ttnn.L1TensorViewPlacement(coord, ttnn.CoreCoord(1, 0), extent.address)
    elif bad == "alignment":
        placement = ttnn.L1TensorViewPlacement(coord, core, extent.address + 1)
    placements = [] if bad == "missing" else [placement]
    if bad == "duplicate":
        placements.append(placement)
    with pytest.raises(RuntimeError):
        ttnn.experimental_create_l1_tensor_view(mesh, spec, placements)


def test_nested_view_reports_only_its_own_extent(per_core_mesh_device):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    view = _view(mesh, spec, owner, [(coord, core, 256)])
    (extent,) = ttnn.experimental_l1_tensor_owned_extents(view)
    assert extent.size == ttnn.experimental_l1_tensor_geometry(mesh, coord, spec).aligned_shard_size
    nested = _view(mesh, _spec([core], 64), view, [(coord, core, 64)])
    (child,) = ttnn.experimental_l1_tensor_owned_extents(nested)
    assert child.address == extent.address + 64
    assert child.size == 64


def test_lockstep_view_requires_one_address(per_core_mesh_device):
    mesh = per_core_mesh_device
    coord = ttnn.MeshCoordinate(0, 0)
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    owner = _owner(mesh, cores[0], per_core=False)
    spec = _spec(cores, 256, per_core=False)
    view = _view(mesh, spec, owner, [(coord, core, 0) for core in cores])
    assert view.buffer_address() == owner.buffer_address()
    with pytest.raises(RuntimeError):
        ttnn.experimental_create_l1_tensor_view(
            mesh,
            spec,
            [ttnn.L1TensorViewPlacement(coord, core, owner.buffer_address() + i * 256) for i, core in enumerate(cores)],
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
    view = _view(mesh, spec, owner, [(coord, core, 0)])
    ttnn.experimental_retain_tensor_backing(view, [owner])
    assert view.device_coords() == [coord]
    assert all(extent.device_coord == coord for extent in ttnn.experimental_l1_tensor_owned_extents(view))
    del owner
    gc.collect()
    torch.testing.assert_close(_read(view), torch.arange(256, dtype=torch.uint8).reshape(1, 256))
    value = torch.full((1, 256), 41, dtype=torch.uint8)
    mapper = ttnn.create_mesh_mapper(
        mesh,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape(1, 1),
            mesh_offset_override=coord,
        ),
    )
    selected_host = ttnn.distribute_tensor(ttnn.from_torch(value, dtype=ttnn.uint8), mapper)
    ttnn.copy_host_to_device_tensor(selected_host, view)
    torch.testing.assert_close(_read(view), value)


def test_allocation_only_views_narrow_writes_and_lifetime(mesh_device):
    """Exercise selected-device allocation, padded writes, semaphore storage and native-copy lifetime."""
    mesh, core = mesh_device, ttnn.CoreCoord(0, 0)
    coord = ttnn.MeshCoordinate(0, 1)
    baseline = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    owner = ttnn.experimental_allocate_single_device_tensor(mesh, coord, _spec([core], 768))
    assert owner.device_coords() == [coord]
    (extent,) = ttnn.experimental_l1_tensor_owned_extents(owner)
    mapper = ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                                  ttnn.MeshShape(1, 1), coord))

    def upload(value, target):
        """Write only the selected device and wait before releasing host data."""
        host = ttnn.distribute_tensor(ttnn.from_torch(value, dtype=ttnn.uint8), mapper)
        ttnn.copy_host_to_device_tensor(host, target)
        ttnn.synchronize_device(mesh)

    upload(torch.full((1, 768), 91, dtype=torch.uint8), owner)
    occupied = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    spec = _spec([core], 17)
    view = _view(mesh, spec, owner, [(coord, core, 256)])
    geometry = ttnn.experimental_l1_tensor_geometry(mesh, coord, spec)
    assert ttnn.experimental_l1_tensor_owned_extents(view)[0].size == geometry.aligned_shard_size
    upload(torch.full((1, 17), 7, dtype=torch.uint8), view)
    submesh = mesh.create_submesh(ttnn.MeshShape(1, 1), coord)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    sem = ttnn.experimental_create_global_semaphore_at_address(submesh, grid, 37, ttnn.BufferType.L1,
                                                             extent.address + 512)
    assert ttnn.get_global_semaphore_address(sem) == extent.address + 512
    del sem
    data = _read(owner).reshape(-1)
    assert torch.all(data[:256] == 91)
    assert torch.all(data[256:273] == 7)
    assert torch.all(data[256 + geometry.aligned_shard_size:512] == 91)
    assert data[512:516].tolist() == [37, 0, 0, 0]
    assert torch.all(data[528:] == 91)
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    ttnn.experimental_retain_tensor_backing(view, [owner])
    copied = ttnn.get_device_tensors(view)[0]
    del owner, view
    gc.collect()
    torch.testing.assert_close(ttnn.to_torch(copied), torch.full((1, 17), 7, dtype=torch.uint8))
    del copied
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline

    owner = ttnn.experimental_allocate_single_device_tensor(mesh, coord, _spec([core], 768))
    view = _view(mesh, _spec([core], 64), owner, [(coord, core, 0)])
    del owner
    gc.collect()
    # Raw views deliberately retain nothing, and destroying one must not free the address again.
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline
    ttnn.deallocate(view)
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline
