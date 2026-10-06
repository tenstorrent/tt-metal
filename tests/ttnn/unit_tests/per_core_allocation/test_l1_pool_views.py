# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import ttnn


def _single_core_spec(core, per_core=True, width=256):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        memory_config.experimental_set_per_core_allocation(True)
    return ttnn.TensorSpec([1, width], ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, memory_config)


def _read_single_device(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def test_l1_pool_view_transfer_guards_and_owner_lifetime(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core)
    geometry = ttnn.experimental_l1_tensor_geometry(per_core_mesh_device, coord, spec)
    free = ttnn.experimental_get_l1_free_ranges(per_core_mesh_device, coord, core)
    owner_size = 3 * geometry.aligned_shard_size
    address = next(start for start, end in free if end - start >= owner_size)

    pool = ttnn.experimental_reserve_l1_pool(
        per_core_mesh_device, [ttnn.L1PoolExtent(coord, core, address, owner_size)]
    )

    def view(slot):
        return ttnn.experimental_create_l1_pool_tensor(
            pool,
            spec,
            [ttnn.L1PoolPlacement(coord, core, 0, slot * geometry.aligned_shard_size)],
        )

    before, payload, after = view(0), view(1), view(2)
    del pool
    gc.collect()

    before_value = torch.full((1, 256), 17, dtype=torch.uint8)
    payload_value = torch.arange(256, dtype=torch.uint8).reshape(1, 256)
    after_value = torch.full((1, 256), 29, dtype=torch.uint8)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(before_value, dtype=ttnn.uint8), before)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(after_value, dtype=ttnn.uint8), after)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(payload_value, dtype=ttnn.uint8), payload)

    torch.testing.assert_close(_read_single_device(before), before_value)
    torch.testing.assert_close(_read_single_device(payload), payload_value)
    torch.testing.assert_close(_read_single_device(after), after_value)


def test_l1_pool_packs_views_at_logical_shard_footprint(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core, width=16)
    geometry = ttnn.experimental_l1_tensor_geometry(per_core_mesh_device, coord, spec)
    logical_size = geometry.aligned_shard_size
    alignment = geometry.allocation_alignment
    owner_size = ((2 * logical_size + alignment - 1) // alignment) * alignment
    address = next(
        start
        for start, end in ttnn.experimental_get_l1_free_ranges(per_core_mesh_device, coord, core)
        if end - start >= owner_size
    )
    pool = ttnn.experimental_reserve_l1_pool(
        per_core_mesh_device, [ttnn.L1PoolExtent(coord, core, address, owner_size)]
    )
    first = ttnn.experimental_create_l1_pool_tensor(
        pool, spec, [ttnn.L1PoolPlacement(coord, core, 0, 0)]
    )
    second = ttnn.experimental_create_l1_pool_tensor(
        pool, spec, [ttnn.L1PoolPlacement(coord, core, 0, logical_size)]
    )
    first_value = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    second_value = first_value + 32
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(first_value, dtype=ttnn.uint8), first)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(second_value, dtype=ttnn.uint8), second)
    torch.testing.assert_close(_read_single_device(first), first_value)
    torch.testing.assert_close(_read_single_device(second), second_value)


def test_lockstep_pool_view_rejects_different_addresses(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[1])])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [1, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    spec = ttnn.TensorSpec([2, 256], ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, memory_config)
    geometry = ttnn.experimental_l1_tensor_geometry(per_core_mesh_device, coord, spec)
    extents = []
    for core in cores:
        address = next(
            start
            for start, end in ttnn.experimental_get_l1_free_ranges(per_core_mesh_device, coord, core)
            if end - start >= 2 * geometry.aligned_shard_size
        )
        extents.append(ttnn.L1PoolExtent(coord, core, address, 2 * geometry.aligned_shard_size))
    pool = ttnn.experimental_reserve_l1_pool(per_core_mesh_device, extents)

    placements = [
        ttnn.L1PoolPlacement(coord, cores[0], 0, 0),
        ttnn.L1PoolPlacement(coord, cores[1], 1, geometry.aligned_shard_size),
    ]
    try:
        ttnn.experimental_create_l1_pool_tensor(pool, spec, placements)
        assert False, "different lockstep addresses must be rejected"
    except RuntimeError as error:
        assert "Lockstep TensorSpec requires one address" in str(error)


def test_l1_pool_adopts_and_retains_existing_tensor(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core, per_core=False, width=16)
    geometry = ttnn.experimental_l1_tensor_geometry(per_core_mesh_device, coord, spec)
    value = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    owner = ttnn.from_torch(
        value,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=per_core_mesh_device,
        memory_config=spec.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
    )
    pool = ttnn.experimental_reserve_l1_pool(per_core_mesh_device, [], external_tensors=[owner])
    assert len(pool.extents) == 1
    assert pool.extents[0].externally_owned
    assert pool.extents[0].size == geometry.allocation_shard_size
    assert geometry.aligned_shard_size <= geometry.allocation_shard_size
    view = ttnn.experimental_create_l1_pool_tensor(
        pool, spec, [ttnn.L1PoolPlacement(coord, core, 0, 0)]
    )
    ttnn.deallocate(owner, force=True)
    del owner
    del pool
    gc.collect()
    torch.testing.assert_close(_read_single_device(view), value)


def test_l1_pool_reservation_failure_rolls_back(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core)
    geometry = ttnn.experimental_l1_tensor_geometry(per_core_mesh_device, coord, spec)
    free_before = list(ttnn.experimental_get_l1_free_ranges(per_core_mesh_device, coord, core))
    address = next(start for start, end in free_before if end - start >= geometry.aligned_shard_size)
    duplicate = ttnn.L1PoolExtent(coord, core, address, geometry.aligned_shard_size)

    try:
        ttnn.experimental_reserve_l1_pool(per_core_mesh_device, [duplicate, duplicate])
        assert False, "overlapping reservations must fail"
    except RuntimeError as error:
        assert "no longer free" in str(error)

    assert list(ttnn.experimental_get_l1_free_ranges(per_core_mesh_device, coord, core)) == free_before


def test_l1_pool_retains_force_deallocated_per_core_owner(per_core_mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core, per_core=True, width=16)
    value = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    owner = ttnn.from_torch(
        value,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=per_core_mesh_device,
        memory_config=spec.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(per_core_mesh_device),
    )
    pool = ttnn.experimental_reserve_l1_pool(per_core_mesh_device, [], external_tensors=[owner])
    view = ttnn.experimental_create_l1_pool_tensor(
        pool, spec, [ttnn.L1PoolPlacement(coord, core, 0, 0)]
    )
    ttnn.deallocate(owner, force=True)
    del owner
    del pool
    gc.collect()
    torch.testing.assert_close(_read_single_device(view), value)


def test_l1_pool_view_uses_only_placement_devices(mesh_device):
    coord = ttnn.MeshCoordinate(0, 0)
    core = ttnn.CoreCoord(0, 0)
    spec = _single_core_spec(core)
    geometry = ttnn.experimental_l1_tensor_geometry(mesh_device, coord, spec)
    address = next(
        start
        for start, end in ttnn.experimental_get_l1_free_ranges(mesh_device, coord, core)
        if end - start >= geometry.aligned_shard_size
    )
    pool = ttnn.experimental_reserve_l1_pool(
        mesh_device, [ttnn.L1PoolExtent(coord, core, address, geometry.aligned_shard_size)]
    )
    view = ttnn.experimental_create_l1_pool_tensor(
        pool, spec, [ttnn.L1PoolPlacement(coord, core, 0, 0)]
    )
    assert list(view.device_coords()) == [coord]
    assert len(ttnn.get_device_tensors(view)) == 1
