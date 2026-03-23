# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for per-core L1 allocation on multi-device (mesh) tensors.

Tests the two-level allocator architecture:
- The mesh-level lockstep allocator lazily queries device per-bank allocators to avoid their regions
- Lockstep allocations from the mesh-level allocator are mirrored into each device's lockstep allocator
- Per-core and lockstep allocations never overlap on any device
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.fixture(scope="function")
def mesh_device(request):
    """Create a mesh device with HYBRID allocator mode for per-core allocation tests."""
    num_devices = ttnn.get_num_devices()
    if num_devices < 2:
        pytest.skip("Multi-device per-core allocation tests require at least 2 devices")

    mesh_shape = ttnn.MeshShape(1, min(num_devices, 2))
    mesh = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        allocator_mode=ttnn.device.AllocatorMode.HYBRID,
    )
    logger.info(f"Opened mesh device with {mesh.get_num_devices()} devices, HYBRID allocator mode")
    yield mesh
    ttnn.close_mesh_device(mesh)


# --- Helpers ---


def _create_per_core_tensor_on_mesh(
    mesh_device, core_grid, shard_shape, tensor_shape, layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
):
    """Create a per-core allocated tensor on a mesh device (replicated).
    tensor_shape: [H, W] shape of the full tensor (must match sharding)."""
    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, shard_spec, per_core_allocation=True)
    data = torch.zeros(*tensor_shape, dtype=torch.uint8)
    return ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _create_lockstep_tensor_on_mesh(
    mesh_device, core_grid, shard_shape, tensor_shape, layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
):
    """Create a lockstep allocated tensor on a mesh device (replicated)."""
    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, shard_spec)
    data = torch.zeros(*tensor_shape, dtype=torch.uint8)
    return ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _single_core_grid(core):
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _create_per_core_single(mesh_device, core, shard_bytes):
    """Shorthand: single-core HEIGHT_SHARDED per-core tensor."""
    return _create_per_core_tensor_on_mesh(mesh_device, _single_core_grid(core), [1, shard_bytes], [1, shard_bytes])


def _create_lockstep_single(mesh_device, core, shard_bytes):
    """Shorthand: single-core HEIGHT_SHARDED lockstep tensor."""
    return _create_lockstep_tensor_on_mesh(mesh_device, _single_core_grid(core), [1, shard_bytes], [1, shard_bytes])


def _addr_ranges_overlap(addr_a, size_a, addr_b, size_b):
    """Check if two address ranges [a, a+size_a) and [b, b+size_b) overlap."""
    return addr_a < addr_b + size_b and addr_b < addr_a + size_a


def _assert_cross_device_consistency(tensor, cores, label=""):
    """Assert all devices have the same per-core addresses."""
    device_tensors = ttnn.get_device_tensors(tensor)
    for core in cores:
        dev_addrs = [dt.per_core_buffer_address(core) for dt in device_tensors]
        assert all(a == dev_addrs[0] for a in dev_addrs), (
            f"{label}Core ({core.x},{core.y}): expected same addr across devices, "
            f"got {[f'{a:#x}' for a in dev_addrs]}"
        )


def _assert_no_overlap_per_device(pc_tensor, pc_core, pc_size, ls_tensor, ls_size, label=""):
    """Assert per-core and lockstep don't overlap on any device."""
    pc_devs = ttnn.get_device_tensors(pc_tensor)
    ls_devs = ttnn.get_device_tensors(ls_tensor)
    for dev_idx, (pc_dt, ls_dt) in enumerate(zip(pc_devs, ls_devs)):
        pc_addr = pc_dt.per_core_buffer_address(pc_core)
        ls_addr = ls_dt.buffer_address()
        assert not _addr_ranges_overlap(pc_addr, pc_size, ls_addr, ls_size), (
            f"{label}Device {dev_idx}: overlap per_core=[{pc_addr:#x}, +{pc_size}) vs "
            f"lockstep=[{ls_addr:#x}, +{ls_size})"
        )


# --- Basic tests ---


def test_per_core_independent_addresses_across_devices(mesh_device):
    """Per-core allocations on different devices get same address when allocator states match."""
    shard_bytes = 2048
    core = ttnn.CoreCoord(0, 0)

    t1 = _create_per_core_single(mesh_device, core, shard_bytes)

    device_tensors = ttnn.get_device_tensors(t1)
    addrs = [dt.per_core_buffer_address(core) for dt in device_tensors]

    assert all(
        a == addrs[0] for a in addrs
    ), f"Expected same initial address on all devices, got {[f'{a:#x}' for a in addrs]}"


def test_per_core_then_lockstep_no_overlap(mesh_device):
    """Per-core first, then lockstep. No overlap on any device."""
    core = ttnn.CoreCoord(0, 0)
    pc_size = 2048
    ls_size = 1024

    t_pc = _create_per_core_single(mesh_device, core, pc_size)
    t_ls = _create_lockstep_single(mesh_device, core, ls_size)

    _assert_no_overlap_per_device(t_pc, core, pc_size, t_ls, ls_size)
    _assert_cross_device_consistency(t_pc, [core])


def test_lockstep_then_per_core_no_overlap(mesh_device):
    """Lockstep first, then per-core. No overlap on any device."""
    core = ttnn.CoreCoord(0, 0)
    ls_size = 2048
    pc_size = 1024

    t_ls = _create_lockstep_single(mesh_device, core, ls_size)
    t_pc = _create_per_core_single(mesh_device, core, pc_size)

    _assert_no_overlap_per_device(t_pc, core, pc_size, t_ls, ls_size)
    _assert_cross_device_consistency(t_pc, [core])


# --- Multi-core tests ---


def test_multi_core_per_core_all_cores(mesh_device):
    """Per-core allocation across all compute cores on mesh, each core independent."""
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]

    SHARD_BYTES = 2048
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    t = _create_per_core_tensor_on_mesh(mesh_device, core_grid, [1, SHARD_BYTES], [num_cores, SHARD_BYTES])

    # All per-device tensors: all cores same address (fresh allocator, independent per-bank)
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(t)):
        addrs = [dt.per_core_buffer_address(c) for c in cores]
        assert all(
            a == addrs[0] for a in addrs
        ), f"Device {dev_idx}: expected same address on all cores, got unique: {set(f'{a:#x}' for a in addrs)}"

    _assert_cross_device_consistency(t, cores)
    assert t.is_allocated()


def test_multi_core_per_core_then_lockstep_no_overlap(mesh_device):
    """Per-core on 4 cores, then lockstep on same 4 cores. No overlap on any device/core."""
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]
    pc_size = 2048
    ls_size = 1024

    pc_tensors = {}
    for i, core in enumerate(cores):
        pc_tensors[i] = _create_per_core_single(mesh_device, core, pc_size)

    ls_tensors = {}
    for i, core in enumerate(cores):
        ls_tensors[i] = _create_lockstep_single(mesh_device, core, ls_size)

    for i, core in enumerate(cores):
        _assert_no_overlap_per_device(pc_tensors[i], core, pc_size, ls_tensors[i], ls_size, label=f"Core {i}: ")
        _assert_cross_device_consistency(pc_tensors[i], [core], label=f"pc core {i}: ")


# --- Interleaved alloc/free patterns ---


def test_interleaved_per_core_and_lockstep(mesh_device):
    """Interleave per-core and lockstep allocations across multiple cores.
    Verify no overlaps on any core on any device."""
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]
    # Track: (label, core_idx, size, kind, tensor)
    live = {}
    tensors = {}

    # Round 1: per-core on each core (varying sizes)
    for i in range(num_cores):
        size = 1024 * (i + 1)
        label = f"pc_c{i}"
        t = _create_per_core_single(mesh_device, cores[i], size)
        tensors[label] = t
        live[label] = (i, size, "per_core")

    # Round 2: lockstep on each core
    for i in range(num_cores):
        size = 2048
        label = f"ls_c{i}"
        t = _create_lockstep_single(mesh_device, cores[i], size)
        tensors[label] = t
        live[label] = (i, size, "lockstep")

    # Round 3: more per-core after lockstep
    for i in range(2):
        size = 512
        label = f"pc2_c{i}"
        t = _create_per_core_single(mesh_device, cores[i], size)
        tensors[label] = t
        live[label] = (i, size, "per_core")

    # Validate no overlaps between per-core and lockstep on same core, per device
    for core_idx in range(num_cores):
        pc_items = [(l, tensors[l], s) for l, (ci, s, k) in live.items() if k == "per_core" and ci == core_idx]
        ls_items = [(l, tensors[l], s) for l, (ci, s, k) in live.items() if k == "lockstep" and ci == core_idx]
        for pc_label, pc_t, pc_size in pc_items:
            for ls_label, ls_t, ls_size in ls_items:
                _assert_no_overlap_per_device(
                    pc_t, cores[core_idx], pc_size, ls_t, ls_size, label=f"{pc_label} vs {ls_label}: "
                )

    # Cross-device consistency for per-core tensors
    for label, (ci, size, kind) in live.items():
        if kind == "per_core":
            _assert_cross_device_consistency(tensors[label], [cores[ci]], label=f"{label}: ")

    for label, t in tensors.items():
        assert t.is_allocated(), f"{label} should still be allocated"


def test_tetris_allocation_on_mesh(mesh_device):
    """Tetris-style alloc/free/realloc across 4 cores on mesh.
    Validates no overlap on same core and cross-device consistency."""
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]

    script = [
        ("c0_a", "alloc", 0, 2048),
        ("c1_a", "alloc", 1, 4096),
        ("c2_a", "alloc", 2, 1024),
        ("c3_a", "alloc", 3, 3072),
        ("c0_b", "alloc", 0, 1024),
        ("c1_b", "alloc", 1, 512),
        ("c2_b", "alloc", 2, 2048),
        ("c0_a", "free", 0, 0),
        ("c1_a", "free", 1, 0),
        ("c0_c", "alloc", 0, 2048),
        ("c1_c", "alloc", 1, 2048),
        ("c1_d", "alloc", 1, 1024),
        ("c3_b", "alloc", 3, 512),
        ("c3_a", "free", 3, 0),
        ("c3_c", "alloc", 3, 3072),
    ]

    tensors = {}
    live_allocs = {}  # label → (core_idx, size)

    for label, action, core_idx, size in script:
        if action == "alloc":
            t = _create_per_core_single(mesh_device, cores[core_idx], size)
            tensors[label] = t
            live_allocs[label] = (core_idx, size)

            # Validate: no overlap with any live allocation on the same core, per device
            for other_label, (other_ci, other_size) in live_allocs.items():
                if other_label != label and other_ci == core_idx:
                    pc_devs = ttnn.get_device_tensors(t)
                    other_devs = ttnn.get_device_tensors(tensors[other_label])
                    for dev_idx, (dt, odt) in enumerate(zip(pc_devs, other_devs)):
                        addr = dt.per_core_buffer_address(cores[core_idx])
                        other_addr = odt.per_core_buffer_address(cores[core_idx])
                        assert not _addr_ranges_overlap(addr, size, other_addr, other_size), (
                            f"Device {dev_idx}: overlap on core {core_idx}: "
                            f"{label}=[{addr:#x}, +{size}) vs {other_label}=[{other_addr:#x}, +{other_size})"
                        )

            # Cross-device consistency
            _assert_cross_device_consistency(t, [cores[core_idx]], label=f"{label}: ")

        elif action == "free":
            del tensors[label]
            del live_allocs[label]

    for label, t in tensors.items():
        assert t.is_allocated(), f"{label} should still be allocated"


# --- Dealloc/realloc tests ---


def test_dealloc_per_core_then_lockstep_reuses_space(mesh_device):
    """Allocate per-core, deallocate, then lockstep gets the same address (space reused)."""
    core = ttnn.CoreCoord(0, 0)
    size = 4096

    t_pc = _create_per_core_single(mesh_device, core, size)
    pc_addr = t_pc.per_core_buffer_address(core)
    t_pc.deallocate(force=True)

    t_ls = _create_lockstep_single(mesh_device, core, size)
    ls_addr = t_ls.buffer_address()
    assert ls_addr == pc_addr, f"Expected lockstep to reuse freed per-core address {pc_addr:#x}, got {ls_addr:#x}"


def test_dealloc_lockstep_then_per_core_reuses_space(mesh_device):
    """Allocate lockstep, deallocate, then per-core gets the same address (space reused)."""
    core = ttnn.CoreCoord(0, 0)
    size = 4096

    t_ls = _create_lockstep_single(mesh_device, core, size)
    ls_addr = t_ls.buffer_address()
    t_ls.deallocate(force=True)

    t_pc = _create_per_core_single(mesh_device, core, size)
    pc_addr = t_pc.per_core_buffer_address(core)
    assert pc_addr == ls_addr, f"Expected per-core to reuse freed lockstep address {ls_addr:#x}, got {pc_addr:#x}"


def test_multi_core_dealloc_realloc(mesh_device):
    """Allocate per-core across all cores, dealloc, realloc — same addresses per device."""
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]

    SHARD_BYTES = 2048
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])

    # First allocation — collect per-device addresses
    t1 = _create_per_core_tensor_on_mesh(mesh_device, core_grid, [1, SHARD_BYTES], [num_cores, SHARD_BYTES])
    addrs1_per_dev = []
    for dt in ttnn.get_device_tensors(t1):
        addrs1_per_dev.append([dt.per_core_buffer_address(c) for c in cores])
    t1.deallocate(force=True)

    # Second allocation — should reuse same addresses on each device
    t2 = _create_per_core_tensor_on_mesh(mesh_device, core_grid, [1, SHARD_BYTES], [num_cores, SHARD_BYTES])
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(t2)):
        addrs2 = [dt.per_core_buffer_address(c) for c in cores]
        assert addrs1_per_dev[dev_idx] == addrs2, (
            f"Device {dev_idx}: expected same addresses after dealloc/realloc:\n"
            f"  first:  {[f'{a:#x}' for a in addrs1_per_dev[dev_idx]]}\n"
            f"  second: {[f'{a:#x}' for a in addrs2]}"
        )


# --- All cores stress test ---


def test_all_cores_lockstep_then_per_core_then_reverse(mesh_device):
    """Phase 1: lockstep → per-core on all cores (no overlap per device).
    Phase 2: deallocate all → per-core → lockstep (deps non-empty)."""
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    shard_bytes = 2048

    # --- Phase 1 ---
    lockstep_tensors = {}
    per_core_tensors = {}

    for i, core in enumerate(cores):
        lockstep_tensors[i] = _create_lockstep_single(mesh_device, core, shard_bytes)

    for i, core in enumerate(cores):
        per_core_tensors[i] = _create_per_core_single(mesh_device, core, shard_bytes)

    for i, core in enumerate(cores):
        _assert_no_overlap_per_device(
            per_core_tensors[i], core, shard_bytes, lockstep_tensors[i], shard_bytes, label=f"Phase 1 core {i}: "
        )

    lockstep_tensors.clear()
    per_core_tensors.clear()

    # --- Phase 2 ---
    pc_sizes = [1024 + (i % 4) * 1024 for i in range(num_cores)]
    for i, core in enumerate(cores):
        per_core_tensors[i] = _create_per_core_single(mesh_device, core, pc_sizes[i])

    for i, core in enumerate(cores):
        lockstep_tensors[i] = _create_lockstep_single(mesh_device, core, shard_bytes)

    for i, core in enumerate(cores):
        _assert_no_overlap_per_device(
            per_core_tensors[i], core, pc_sizes[i], lockstep_tensors[i], shard_bytes, label=f"Phase 2 core {i}: "
        )

    for i in range(num_cores):
        assert lockstep_tensors[i].is_allocated()
        assert per_core_tensors[i].is_allocated()


# --- Triangle pattern (proves independent per-bank state) ---


def test_triangle_per_core_then_uniform(mesh_device):
    """Triangle per-core on all cores, then uniform per-core.
    Addresses form inverse triangle, consistent across devices, no overlap."""
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    num_cores = len(cores)
    TILE_BYTES = 1024

    # Phase 1: Triangle — core i gets (i+1) * TILE_BYTES
    triangle_tensors = {}
    for i, core in enumerate(cores):
        size = (i + 1) * TILE_BYTES
        triangle_tensors[i] = _create_per_core_single(mesh_device, core, size)

    # Phase 2: Uniform per-core across all cores
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sharded_tensor = _create_per_core_tensor_on_mesh(mesh_device, core_grid, [1, TILE_BYTES], [num_cores, TILE_BYTES])

    # Validate inverse triangle per device
    for dev_idx, dt in enumerate(ttnn.get_device_tensors(sharded_tensor)):
        addrs = [dt.per_core_buffer_address(c) for c in cores]
        for i in range(num_cores - 1):
            assert addrs[i] > addrs[i + 1], (
                f"Device {dev_idx}: expected inverse triangle: "
                f"core {i} addr={addrs[i]:#x} > core {i+1} addr={addrs[i+1]:#x}"
            )

    # Cross-device consistency
    _assert_cross_device_consistency(sharded_tensor, cores, label="uniform: ")
    for i, core in enumerate(cores):
        _assert_cross_device_consistency(triangle_tensors[i], [core], label=f"triangle[{i}]: ")

    # No overlap between triangle and uniform on same core, per device
    for i, core in enumerate(cores):
        tri_size = (i + 1) * TILE_BYTES
        tri_devs = ttnn.get_device_tensors(triangle_tensors[i])
        uni_devs = ttnn.get_device_tensors(sharded_tensor)
        for dev_idx, (tri_dt, uni_dt) in enumerate(zip(tri_devs, uni_devs)):
            tri_addr = tri_dt.per_core_buffer_address(core)
            uni_addr = uni_dt.per_core_buffer_address(core)
            assert not _addr_ranges_overlap(tri_addr, tri_size, uni_addr, TILE_BYTES), (
                f"Device {dev_idx} core {i}: triangle=[{tri_addr:#x}, +{tri_size}) vs "
                f"uniform=[{uni_addr:#x}, +{TILE_BYTES})"
            )

    for i in range(num_cores):
        assert triangle_tensors[i].is_allocated()
    assert sharded_tensor.is_allocated()


# --- Divergent per-device state ---


def _allocate_per_core_on_device(mesh_device, coord, core, shard_bytes):
    """Create a per-core tensor on a single device within the mesh.
    Creates a host tensor and writes it to the target device's allocator.
    The mesh-level lockstep allocator lazily queries device state, so no mirroring needed."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        [1, shard_bytes],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
        per_core_allocation=True,
    )
    data = torch.arange(shard_bytes, dtype=torch.uint8).reshape(1, shard_bytes)
    host_tensor = ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn._ttnn.multi_device.to_single_device(host_tensor, mesh_device, coord, mem_config)


def test_single_device_loopback(mesh_device):
    """Write data to a single device within the mesh and read it back.
    Verifies the full round-trip: host → single device → host."""
    core = ttnn.CoreCoord(0, 0)
    shard_bytes = 2048

    for dev_idx in range(mesh_device.get_num_devices()):
        coord = ttnn.MeshCoordinate(0, dev_idx)
        t = _allocate_per_core_on_device(mesh_device, coord, core, shard_bytes)

        # Read back and verify
        result = ttnn.to_torch(ttnn.from_device(t))
        expected = torch.arange(shard_bytes, dtype=torch.uint8).reshape(1, shard_bytes)
        assert torch.equal(result, expected), (
            f"Device {dev_idx}: loopback data mismatch. "
            f"Expected first 8: {expected[0,:8].tolist()}, got: {result[0,:8].tolist()}"
        )


def test_divergent_per_device_then_lockstep(mesh_device):
    """Allocate different-sized per-core tensors on each device independently,
    causing divergent allocator states. Then lockstep must get the same address
    on all devices and not overlap with any per-core allocation.

    This simulates the compressed tensor TP scenario where each device's
    weight slice compresses to different sizes.
    """
    core = ttnn.CoreCoord(0, 0)
    dev0_coord = ttnn.MeshCoordinate(0, 0)
    dev1_coord = ttnn.MeshCoordinate(0, 1)

    # Allocate different sizes per device to diverge allocator states
    per_device_sizes = [2048, 4096]
    t0 = _allocate_per_core_on_device(mesh_device, dev0_coord, core, per_device_sizes[0])
    t1 = _allocate_per_core_on_device(mesh_device, dev1_coord, core, per_device_sizes[1])

    addr0 = t0.per_core_buffer_address(core)
    addr1 = t1.per_core_buffer_address(core)
    logger.info(f"Device 0: per-core size={per_device_sizes[0]}, addr={addr0:#x}")
    logger.info(f"Device 1: per-core size={per_device_sizes[1]}, addr={addr1:#x}")

    # Top-down allocator: addr = L1_TOP - size. addr0 - addr1 = size1 - size0
    expected_diff = per_device_sizes[1] - per_device_sizes[0]
    actual_diff = addr0 - addr1
    assert actual_diff == expected_diff, (
        f"Expected addr difference {expected_diff}, got {actual_diff}. "
        f"dev0={addr0:#x} (size={per_device_sizes[0]}), dev1={addr1:#x} (size={per_device_sizes[1]})"
    )

    # Lockstep allocation — mirroring ensures it avoids both per-core regions
    ls_size = 1024
    t_ls = _create_lockstep_single(mesh_device, core, ls_size)

    # Lockstep must be same address on all devices
    ls_devs = ttnn.get_device_tensors(t_ls)
    ls_addrs = [dt.buffer_address() for dt in ls_devs]
    assert all(
        a == ls_addrs[0] for a in ls_addrs
    ), f"Lockstep should be same on all devices, got {[f'{a:#x}' for a in ls_addrs]}"

    # No overlap with per-core on either device
    assert not _addr_ranges_overlap(
        addr0, per_device_sizes[0], ls_addrs[0], ls_size
    ), f"Device 0: overlap per_core=[{addr0:#x}, +{per_device_sizes[0]}) vs lockstep=[{ls_addrs[0]:#x}, +{ls_size})"
    assert not _addr_ranges_overlap(
        addr1, per_device_sizes[1], ls_addrs[1], ls_size
    ), f"Device 1: overlap per_core=[{addr1:#x}, +{per_device_sizes[1]}) vs lockstep=[{ls_addrs[1]:#x}, +{ls_size})"


def test_divergent_multi_core_per_device_then_lockstep(mesh_device):
    """Multiple cores, different sizes per device, then lockstep.
    Simulates TP compressed weights across cores."""
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]
    dev0_coord = ttnn.MeshCoordinate(0, 0)
    dev1_coord = ttnn.MeshCoordinate(0, 1)

    # Device 0: increasing sizes per core. Device 1: decreasing sizes.
    dev0_sizes = [1024, 2048, 3072, 4096]
    dev1_sizes = [4096, 3072, 2048, 1024]

    dev0_tensors = []
    dev1_tensors = []
    dev0_addrs = []
    dev1_addrs = []

    for core_idx, core in enumerate(cores):
        t0 = _allocate_per_core_on_device(mesh_device, dev0_coord, core, dev0_sizes[core_idx])
        t1 = _allocate_per_core_on_device(mesh_device, dev1_coord, core, dev1_sizes[core_idx])
        dev0_tensors.append(t0)
        dev1_tensors.append(t1)
        dev0_addrs.append(t0.per_core_buffer_address(core))
        dev1_addrs.append(t1.per_core_buffer_address(core))

    # Verify address differences match size differences
    for core_idx in range(num_cores):
        expected_diff = dev1_sizes[core_idx] - dev0_sizes[core_idx]
        actual_diff = dev0_addrs[core_idx] - dev1_addrs[core_idx]
        assert actual_diff == expected_diff, (
            f"Core {core_idx}: expected addr diff {expected_diff}, got {actual_diff}. "
            f"dev0={dev0_addrs[core_idx]:#x} (size={dev0_sizes[core_idx]}), "
            f"dev1={dev1_addrs[core_idx]:#x} (size={dev1_sizes[core_idx]})"
        )

    # Lockstep on each core — must be same across devices, no overlap on either
    ls_size = 512
    for core_idx, core in enumerate(cores):
        t_ls = _create_lockstep_single(mesh_device, core, ls_size)

        # Same address on all devices
        ls_devs = ttnn.get_device_tensors(t_ls)
        ls_addrs = [dt.buffer_address() for dt in ls_devs]
        assert all(
            a == ls_addrs[0] for a in ls_addrs
        ), f"Core {core_idx}: lockstep not same across devices: {[f'{a:#x}' for a in ls_addrs]}"

        # No overlap with per-core on either device
        assert not _addr_ranges_overlap(dev0_addrs[core_idx], dev0_sizes[core_idx], ls_addrs[0], ls_size), (
            f"Device 0 core {core_idx}: overlap "
            f"per_core=[{dev0_addrs[core_idx]:#x}, +{dev0_sizes[core_idx]}) vs lockstep=[{ls_addrs[0]:#x}, +{ls_size})"
        )
        assert not _addr_ranges_overlap(dev1_addrs[core_idx], dev1_sizes[core_idx], ls_addrs[1], ls_size), (
            f"Device 1 core {core_idx}: overlap "
            f"per_core=[{dev1_addrs[core_idx]:#x}, +{dev1_sizes[core_idx]}) vs lockstep=[{ls_addrs[1]:#x}, +{ls_size})"
        )
