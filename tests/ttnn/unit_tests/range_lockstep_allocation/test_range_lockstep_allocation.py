# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests: a MemoryConfig asking for range lockstep must reach the allocator.

No link between MemoryConfig and the allocator errors when it drops the flag; the request just
degrades to default lockstep. So these assert on allocation outcome -- whether a placement
succeeds beside a per-core allocation -- rather than reading the flag back.

Mesh, not a plain device: hybrid_device_allocators_ is only populated for mesh allocations, and
the ranges are only gathered when it is non-empty. A 1x1 mesh is enough.
"""

import pytest
import torch

import ttnn


PAGE_SIZE = 32

HOGGED_CORE = ttnn.CoreCoord(0, 0)
FREE_CORE = ttnn.CoreCoord(1, 0)


def _num_cores(first_core, last_core):
    """Cores in the inclusive rectangle first_core..last_core -- a CoreRange is an area, not a row.

    ttnn.CoreRange rejects a reversed range, but only once the config is built, and this runs
    first to size the tensor -- where reversing both dimensions multiplies two negatives back
    into a plausible count.
    """
    assert (
        first_core.x <= last_core.x and first_core.y <= last_core.y
    ), f"core range is reversed: ({first_core.x},{first_core.y})..({last_core.x},{last_core.y})"
    return (last_core.x - first_core.x + 1) * (last_core.y - first_core.y + 1)


def _grid_config(first_core, last_core, num_bytes, *, range_lockstep=False, per_core=False):
    """Height-sharded L1 config giving every core in first_core..last_core ``num_bytes``.

    The shard shape has to match the tensor's own width; TensorSpec validates that before the
    allocator is ever reached.
    """
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(first_core, last_core)]),
            [1, num_bytes],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    if per_core:
        mem_config.experimental_set_per_core_allocation(True)
    if range_lockstep:
        mem_config.experimental_set_range_lockstep_allocation(True)
    return mem_config


def _allocate(mesh, first_core, num_bytes, *, last_core=None, range_lockstep=False, per_core=False):
    """Allocate ``num_bytes`` on every core of the rectangle first_core..last_core.

    One row of the tensor per core, so the shard shape stays [1, num_bytes] whatever the grid.
    """
    last_core = first_core if last_core is None else last_core
    data = torch.zeros(_num_cores(first_core, last_core), num_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=_grid_config(first_core, last_core, num_bytes, range_lockstep=range_lockstep, per_core=per_core),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
    )


def _hog_size(mesh):
    """Over half of a bank's largest free block, so two of them cannot both be placed.

    Read from the live allocator because bank size varies with architecture and harvesting.
    """
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    largest = view.largest_contiguous_bytes_free_per_bank
    num_bytes = (largest * 6 // 10) // PAGE_SIZE * PAGE_SIZE
    assert num_bytes > largest // 2, "sized under half of L1; both allocations would fit even under a chip-wide scan"
    return num_bytes


@pytest.fixture
def hog(hybrid_mesh_device):
    """Factory: occupy most of one core's L1 with a per-core allocation.

    Yields (mesh, place), where place(core) -> the size the test should then request. Held for
    the body of the test and released after.
    """
    mesh = hybrid_mesh_device
    holder = []

    def place(core):
        num_bytes = _hog_size(mesh)
        holder.append(_allocate(mesh, core, num_bytes, per_core=True))
        return num_bytes

    yield mesh, place
    holder.clear()


def _require_cores(mesh, count):
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < count:
        pytest.skip(f"need {count} compute cores in a row, grid is {grid.x}x{grid.y}")


@pytest.fixture
def hogged_mesh(hog):
    """The common case: HOGGED_CORE occupied, yielding (mesh, num_bytes)."""
    mesh, place = hog
    _require_cores(mesh, 2)
    yield mesh, place(HOGGED_CORE)


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


@pytest.mark.parametrize("hog_index", [0, 1, 2], ids=["hog-first", "hog-middle", "hog-last"])
def test_range_lockstep_still_avoids_per_core_ranges_on_its_own_cores(hog, expect_error, hog_index):
    """Scoping the scan must narrow it to the buffer's own cores, not switch it off.

    The grid covers the hogged core, so that occupancy is the buffer's own business and must
    still block the placement. Every other test here passes against a scan that collects nothing.

    Each hog position rules out a different wrong scan: collecting nothing fails all three, only
    the grid's first core fails middle and last, dropping the grid's last core fails last.
    """
    mesh, place = hog
    _require_cores(mesh, 3)
    first, last = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0)
    num_bytes = place(ttnn.CoreCoord(hog_index, 0))
    with expect_error(RuntimeError, "Out of Memory"):
        _allocate(mesh, first, num_bytes, last_core=last, range_lockstep=True)


def test_range_lockstep_spans_several_free_cores(hogged_mesh):
    """A multi-core grid that excludes the hogged core must still be placeable."""
    mesh, num_bytes = hogged_mesh
    _require_cores(mesh, 4)
    tensor = _allocate(mesh, FREE_CORE, num_bytes, last_core=ttnn.CoreCoord(3, 0), range_lockstep=True)
    assert tensor is not None


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
        memory_config=_grid_config(HOGGED_CORE, HOGGED_CORE, num_bytes, range_lockstep=True),
        mesh_mapper=ttnn.ReplicateTensorToMesh(hybrid_mesh_device),
        device=hybrid_mesh_device,
    )
    result = ttnn.to_torch(ttnn.from_device(tensor), mesh_composer=ttnn.ConcatMeshToTensor(hybrid_mesh_device, dim=0))
    assert torch.equal(data, result), "round-trip data mismatch"
