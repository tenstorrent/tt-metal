# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import torch

import ttnn

# Views reference caller-owned SRAM; all owning allocations stay explicit in these tests.


def _spec(cores, width, per_core=True):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        memory.experimental_set_per_core_allocation(True)
    spec = ttnn.TensorSpec([len(cores), width], ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, memory)
    assert spec.memory_config.experimental_is_per_core_allocation() == per_core
    return spec


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
    """Describe the remaining owned bytes after each view offset."""
    core_count = owner.memory_config().shard_spec.grid.num_cores()
    size = owner.spec.compute_consumed_memory_bytes_per_bank(ttnn.get_l1_alignment(), core_count)
    spans = _memory(
        [(coord, core, _address(owner, coord, core) + offset, size - offset) for coord, core, offset in placements]
    )
    return ttnn.experimental_create_sram_tensor_view(mesh, spec, spans)


def _memory(entries):
    """Build explicit per-device and per-core spans for view tests."""
    devices = {}
    for coord, core, address, size in entries:
        devices.setdefault(coord, {})[core] = ttnn.NodeLocalSpan(address, size)
    return ttnn.PerNodeMeshDeviceMemorySpan(
        {coord: ttnn.PerNodeDeviceLocalMemorySpan(cores) for coord, cores in devices.items()}
    )


def _address(tensor, coord, core):
    """Read the tensor's existing address API without inspecting its owner."""
    if tensor.is_per_core_allocated():
        return tensor.experimental_per_core_buffer_address(coord, core)
    return tensor.buffer_address()


@pytest.mark.parametrize("per_core", [True, False])
def test_caller_owned_views_preserve_neighbor_bytes(per_core_mesh_device, per_core):
    """Keep backing alive explicitly; destroying views must not release its bytes."""
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    baseline = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    owner = _owner(mesh, core, per_core=per_core)
    spec = _spec([core], 256)
    before = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    views = [_view(mesh, spec, owner, [(coord, core, offset)]) for offset in (0, 256, 512)]
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == before
    values = [torch.full((1, 256), value, dtype=torch.uint8) for value in (17, 23, 29)]
    for view, value in zip(views, values):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=ttnn.uint8), view)
    for view, value in zip(views, values):
        torch.testing.assert_close(_read(view), value)
    del view, views
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == before
    del owner
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline


@pytest.mark.parametrize("bad", ["alignment", "short_span", "missing"])
def test_view_rejects_invalid_placements(per_core_mesh_device, bad, expect_error):
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    address = _address(owner, coord, core)
    placement = (coord, core, address, 256)
    if bad == "alignment":
        placement = (coord, core, address + 1, 256)
    if bad == "short_span":
        placement = (coord, core, address, 255)
    placements = [] if bad == "missing" else [placement]
    with expect_error(RuntimeError, "SRAM"):
        ttnn.experimental_create_sram_tensor_view(mesh, spec, _memory(placements))
    # Expected-error tracebacks can retain the owner past device teardown; release its storage now.
    ttnn.deallocate(owner)


def test_view_rejects_span_that_fits_payload_but_not_padding(per_core_mesh_device, expect_error):
    """Reject a 17-byte span when the tensor's aligned writes need more bytes."""
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 17)
    footprint = spec.compute_consumed_memory_bytes_per_bank(ttnn.get_l1_alignment(), 1)
    assert footprint > 17
    with expect_error(RuntimeError, "padded tensor footprint exceeds supplied span"):
        ttnn.experimental_create_sram_tensor_view(
            mesh, spec, _memory([(coord, core, _address(owner, coord, core), 17)])
        )
    view = ttnn.experimental_create_sram_tensor_view(
        mesh, spec, _memory([(coord, core, _address(owner, coord, core), footprint)])
    )
    assert _address(view, coord, core) == _address(owner, coord, core)
    assert torch.all(_read(owner) == 0)
    ttnn.deallocate(view)
    ttnn.deallocate(owner)


def test_nested_view_preserves_address_and_geometry(per_core_mesh_device):
    """Nested views expose their own spec and offset without changing their backing."""
    mesh = per_core_mesh_device
    coord, core = ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    spec = _spec([core], 256)
    view = _view(mesh, spec, owner, [(coord, core, 256)])
    assert view.spec == spec
    assert _address(view, coord, core) == _address(owner, coord, core) + 256
    nested = _view(mesh, _spec([core], 64), view, [(coord, core, 64)])
    assert _address(nested, coord, core) == _address(owner, coord, core) + 320
    assert nested.spec.compute_consumed_memory_bytes_per_bank(ttnn.get_l1_alignment(), 1) == 64


def test_lockstep_view_requires_one_address(per_core_mesh_device, expect_error):
    mesh = per_core_mesh_device
    coord = ttnn.MeshCoordinate(0, 0)
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    owner = _owner(mesh, cores[0], per_core=False)
    spec = _spec(cores, 256, per_core=False)
    view = _view(mesh, spec, owner, [(coord, core, 0) for core in cores])
    assert view.buffer_address() == owner.buffer_address()
    with expect_error(RuntimeError, "Lockstep tensor view requires one address"):
        ttnn.experimental_create_sram_tensor_view(
            mesh,
            spec,
            _memory([(coord, core, owner.buffer_address() + i * 256, 256) for i, core in enumerate(cores)]),
        )
    ttnn.deallocate(view)
    ttnn.deallocate(owner)


def test_distinct_per_core_addresses_preserve_backing(per_core_mesh_device, expect_error):
    """Exercise different addresses on two cores, including padding and untouched neighbors."""
    mesh, coord = per_core_mesh_device, ttnn.MeshCoordinate(0, 0)
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    owner_spec = _spec(cores, 768)
    initial = torch.full((2, 768), 91, dtype=torch.uint8)
    owner = ttnn.from_torch(
        initial,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=owner_spec.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    spec = _spec(cores, 256)
    with expect_error(RuntimeError, "SRAM view placements must cover every configured data core"):
        _view(mesh, spec, owner, [(coord, cores[0], 0)])
    view = _view(mesh, spec, owner, [(coord, cores[0], 0), (coord, cores[1], 256)])
    addresses = [_address(view, coord, core) for core in cores]
    assert addresses[0] != addresses[1]
    values = torch.stack([torch.full((256,), value, dtype=torch.uint8) for value in (17, 29)])
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(values, dtype=ttnn.uint8), view)
    torch.testing.assert_close(_read(view), values)
    expected = initial.clone()
    expected[0, :256] = 17
    expected[1, 256:512] = 29
    del view
    gc.collect()
    torch.testing.assert_close(_read(owner), expected)
    ttnn.deallocate(owner)


def test_view_preserves_selected_device_coverage(mesh_device):
    mesh = mesh_device
    coord, core = ttnn.MeshCoordinate(0, 1), ttnn.CoreCoord(0, 0)
    spec = _spec([core], 256)
    host = ttnn.from_torch(torch.arange(256, dtype=torch.uint8).reshape(1, 256), dtype=ttnn.uint8)
    owner = ttnn._ttnn.tensor.experimental_to_single_device(host, mesh, coord, spec.memory_config)
    view = _view(mesh, spec, owner, [(coord, core, 0)])
    assert view.device_coords() == [coord]
    assert _address(view, coord, core) == _address(owner, coord, core)
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


def test_view_device_order_is_independent_of_span_insertion(mesh_device):
    """Keep tensor topology and reference addresses stable for unordered span maps."""
    mesh, core = mesh_device, ttnn.CoreCoord(0, 0)
    # Copy iterator values because the native range iterator reuses its coordinate object.
    coords = [ttnn.MeshCoordinate(list(coord)) for coord in ttnn.MeshCoordinateRange(mesh.shape)][:2]
    assert len(coords) == 2
    owner = _owner(mesh, core)
    entries = [(coord, core, _address(owner, coord, core), 256) for coord in coords]
    first = ttnn.experimental_create_sram_tensor_view(mesh, _spec([core], 256), _memory(entries))
    second = ttnn.experimental_create_sram_tensor_view(mesh, _spec([core], 256), _memory(reversed(entries)))
    assert first.device_coords() == second.device_coords() == coords
    for coord in coords:
        assert _address(first, coord, core) == _address(second, coord, core) == _address(owner, coord, core)


def test_allocation_only_views_narrow_writes_and_lifetime(mesh_device):
    """Exercise selected-device allocation, narrow writes and caller-owned backing for native copies."""
    mesh, core = mesh_device, ttnn.CoreCoord(0, 0)
    coord, other = ttnn.MeshCoordinate(0, 1), ttnn.MeshCoordinate(0, 0)
    other_before = ttnn.experimental_get_l1_occupied_ranges(mesh, other, core)
    baseline = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    owner_spec = _spec([core], 768)
    owner = ttnn.experimental_allocate_single_device_tensor(mesh, coord, owner_spec)
    assert owner.device_coords() == [coord]
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, other, core) == other_before
    address = _address(owner, coord, core)
    mapper = ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 1), coord)
    )

    def upload(value, target):
        """Write only the selected device and wait before releasing host data."""
        host = ttnn.distribute_tensor(ttnn.from_torch(value, dtype=ttnn.uint8), mapper)
        ttnn.copy_host_to_device_tensor(host, target)
        ttnn.synchronize_device(mesh)

    upload(torch.full((1, 768), 91, dtype=torch.uint8), owner)
    occupied = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    spec = _spec([core], 17)
    view = _view(mesh, spec, owner, [(coord, core, 256)])
    footprint = spec.compute_consumed_memory_bytes_per_bank(ttnn.get_l1_alignment(), 1)
    assert view.spec == spec
    upload(torch.full((1, 17), 7, dtype=torch.uint8), view)
    data = _read(owner).reshape(-1)
    assert torch.all(data[:256] == 91)
    assert torch.all(data[256:273] == 7)
    assert torch.all(data[256 + footprint :] == 91)
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    copied = ttnn.get_device_tensors(view)[0]
    del view
    gc.collect()
    torch.testing.assert_close(ttnn.to_torch(copied), torch.full((1, 17), 7, dtype=torch.uint8))
    del copied
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    del owner
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline

    owner = ttnn.experimental_allocate_single_device_tensor(mesh, coord, owner_spec)
    view = _view(mesh, _spec([core], 64), owner, [(coord, core, 0)])
    assert _address(owner, coord, core) == address
    # Reacquisition must not zero or upload over the bytes written by the previous owner.
    torch.testing.assert_close(_read(owner).reshape(-1), data)
    del owner
    gc.collect()
    # Raw views deliberately retain nothing, and destroying one must not free the address again.
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline
    ttnn.deallocate(view)
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == baseline
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, other, core) == other_before


def test_semaphore_rejects_invalid_addresses_before_writing(per_core_mesh_device, expect_error):
    """Reject reserved, unaligned and out-of-range semaphore writes without touching backing bytes."""
    mesh, coord, core = per_core_mesh_device, ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)
    owner = _owner(mesh, core)
    address = _address(owner, coord, core)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    for invalid, message in (
        (0, "outside allocatable L1"),
        (address + 1, "must be aligned"),
        (2**32, "outside allocatable L1"),
        (2**64 - ttnn.get_l1_alignment(), "outside allocatable L1"),
    ):
        with expect_error(RuntimeError, message):
            ttnn.experimental_create_global_semaphore_at_address(mesh, grid, 37, ttnn.BufferType.L1, invalid)
    with expect_error(RuntimeError, "ordinary SRAM"):
        ttnn.experimental_create_global_semaphore_at_address(mesh, grid, 37, ttnn.BufferType.L1_SMALL, address)
    assert torch.all(_read(owner) == 0)
    occupied = ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core)
    # The semaphore writes every supplied device, so use a single-device fixture with owned backing.
    sem = ttnn.experimental_create_global_semaphore_at_address(mesh, grid, 37, ttnn.BufferType.L1, address + 256)
    assert ttnn.get_global_semaphore_address(sem) == address + 256
    data = _read(owner).reshape(-1)
    assert torch.all(data[:256] == 0)
    assert data[256:260].tolist() == [37, 0, 0, 0]
    assert torch.all(data[256 + ttnn.get_l1_alignment() :] == 0)
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    del sem
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    gc.collect()
    assert ttnn.experimental_get_l1_occupied_ranges(mesh, coord, core) == occupied
    torch.testing.assert_close(_read(owner).reshape(-1), data)
    ttnn.deallocate(owner)
