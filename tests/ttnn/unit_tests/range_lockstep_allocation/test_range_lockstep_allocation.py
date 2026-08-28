# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests: a MemoryConfig asking for range lockstep must reach the allocator.

The host-side suite next door proves the flag survives MemoryConfig's identity, equality and
JSON. This one proves the whole chain works on a device -- MemoryConfig -> TensorSpec ->
TensorLayout::compute_buffer_sharding_args -> BufferShardingArgs -> Buffer -> allocator -- by
observing the only externally visible difference the mode makes: whether a placement succeeds
next to a per-core allocation on a core the buffer does not occupy.

Nothing in the chain errors when a link drops the flag; the request just silently degrades to
default lockstep. That is why this is tested by allocation outcome rather than by reading the
flag back.

Mesh, not a plain device: hybrid_device_allocators_ is only populated for mesh allocations, and
the per-core ranges are only gathered when it is non-empty. A 1x1 mesh is enough.
"""

import pytest
import torch

import ttnn


PAGE_SIZE = 32

HOGGED_CORE = ttnn.CoreCoord(0, 0)
FREE_CORE = ttnn.CoreCoord(1, 0)


def _single_core_config(core, num_bytes, *, range_lockstep=False, per_core=False):
    """Height-sharded L1 config putting all ``num_bytes`` on exactly ``core``.

    The shard shape has to match the tensor's own width; TensorSpec validates that before the
    allocator is ever reached.
    """
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            [1, num_bytes],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    if per_core:
        mem_config.experimental_set_per_core_allocation(True)
    if range_lockstep:
        mem_config.experimental_set_range_lockstep_allocation(True)
    return mem_config


def _allocate(mesh, core, num_bytes, *, range_lockstep=False, per_core=False):
    data = torch.zeros(1, num_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=_single_core_config(core, num_bytes, range_lockstep=range_lockstep, per_core=per_core),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
    )


def _hog_size(mesh):
    """Over half of a bank's largest free block, so two of them cannot both be placed.

    Sized from the live allocator rather than a constant: the bank size differs across
    architectures and harvesting, and a fixed number would either not fill half of L1 (making
    the negative test pass for the wrong reason) or not fit at all.
    """
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    largest = view.largest_contiguous_bytes_free_per_bank
    num_bytes = (largest * 6 // 10) // PAGE_SIZE * PAGE_SIZE
    assert num_bytes > largest // 2, "sized under half of L1; both allocations would fit even under a chip-wide scan"
    return num_bytes


@pytest.fixture
def hogged_mesh(hybrid_mesh_device):
    """A mesh whose core (0,0) holds a per-core allocation covering most of its L1.

    Yields (mesh, num_bytes) where num_bytes is the same size the test should then request on
    FREE_CORE. The hog is kept alive for the body of the test and released after.
    """
    mesh = hybrid_mesh_device
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 2:
        pytest.skip(f"need two disjoint compute cores in a row, grid is {grid.x}x{grid.y}")

    num_bytes = _hog_size(mesh)
    hog = _allocate(mesh, HOGGED_CORE, num_bytes, per_core=True)
    yield mesh, num_bytes
    del hog


def test_range_lockstep_ignores_per_core_ranges_on_other_cores(hogged_mesh):
    """FREE_CORE holds nothing, so a range lockstep buffer of the same size must fit there."""
    mesh, num_bytes = hogged_mesh
    tensor = _allocate(mesh, FREE_CORE, num_bytes, range_lockstep=True)
    assert tensor is not None


def test_default_lockstep_still_avoids_per_core_ranges_everywhere(hogged_mesh, expect_error):
    """The default must keep scanning every bank.

    This is the half that makes a multicast past its own cores safe, so it has to stay
    expensive. If this test starts passing an allocation, the opt-in stopped being opt-in.
    """
    mesh, num_bytes = hogged_mesh
    with expect_error(RuntimeError, "Out of Memory"):
        _allocate(mesh, FREE_CORE, num_bytes)


def test_range_lockstep_round_trip(hybrid_mesh_device):
    """Data written through a range lockstep config reads back intact.

    Scoping the range scan must not disturb where the buffer actually lands.
    """
    num_bytes = 2048
    data = torch.arange(num_bytes, dtype=torch.uint8).reshape(1, num_bytes)
    tensor = ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=_single_core_config(HOGGED_CORE, num_bytes, range_lockstep=True),
        mesh_mapper=ttnn.ReplicateTensorToMesh(hybrid_mesh_device),
        device=hybrid_mesh_device,
    )
    result = ttnn.to_torch(ttnn.from_device(tensor), mesh_composer=ttnn.ConcatMeshToTensor(hybrid_mesh_device, dim=0))
    assert torch.equal(data, result), "round-trip data mismatch"
