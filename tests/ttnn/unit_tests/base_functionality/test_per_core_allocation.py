# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for per-core L1 allocation via MemoryConfig.per_core_shard_sizes.

Each test creates single-core HEIGHT_SHARDED tensors with per_core_shard_sizes=[size],
which triggers the per-bank allocator (AllocatorID bank_id+1) instead of lockstep.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


class PerCoreMemMap:
    """Track expected per-core allocator addresses for test validation.

    The L1 per-bank allocator is TOP-DOWN: allocations grow from the top of
    the L1 address space downward. Each core has an independent free-list
    starting from the same top address.

    Derive L1_TOP from the first allocation: L1_TOP = first_addr + first_size.
    """

    def __init__(self, first_addr, first_size):
        self.l1_top = first_addr + first_size
        self.expected = {}
        # Per-core: list of (addr, size) — active allocations
        self._core_allocs = {}

    def alloc(self, label, core_id, size):
        """Record a top-down allocation and compute expected address.

        Scans from top downward, finds first gap that fits.
        """
        if core_id not in self._core_allocs:
            self._core_allocs[core_id] = []

        allocs = self._core_allocs[core_id]
        # Sort by address descending (top-down scan)
        sorted_allocs = sorted(allocs, key=lambda x: x[0], reverse=True)

        # Try to fit in gaps between allocations (scanning top-down)
        candidate = self.l1_top - size
        for a_addr, a_size in sorted_allocs:
            a_end = a_addr + a_size
            if candidate >= a_end:
                # Fits above this allocation
                break
            # Blocked by this allocation, try below it
            candidate = a_addr - size

        allocs.append((candidate, size))
        self.expected[label] = candidate
        return candidate

    def free(self, core_id, addr):
        """Record a deallocation."""
        self._core_allocs[core_id] = [(a, s) for a, s in self._core_allocs[core_id] if a != addr]

    def validate(self, actual):
        """Assert actual address map matches expected."""
        assert actual == self.expected, f"Address map mismatch:\n" + "\n".join(
            f"  {k}: expected={self.expected[k]:#x} actual={actual.get(k, 'MISSING'):#x}"
            f"{' MISMATCH' if self.expected[k] != actual.get(k) else ''}"
            for k in self.expected
        )


def _create_single_core_tensor(device, core, shard_bytes):
    """Create a single-core HEIGHT_SHARDED L1 tensor with per-core allocation."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        [1, shard_bytes],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
        per_core_shard_sizes=[shard_bytes],
    )
    data = torch.zeros(1, shard_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )


def test_per_core_tensors_get_same_address(device):
    """Two single-core tensors on different cores can share the same L1 address,
    proving they use independent per-bank allocators (not lockstep)."""
    shard_bytes = 1024
    core0 = ttnn.CoreCoord(0, 0)
    core1 = ttnn.CoreCoord(1, 0)

    t0 = _create_single_core_tensor(device, core0, shard_bytes)
    t1 = _create_single_core_tensor(device, core1, shard_bytes)

    # Per-bank allocators are independent — both cores can get the same address
    # With lockstep, t1 would get a different address since t0 consumed it
    addr0 = t0.buffer_address()
    addr1 = t1.buffer_address()
    assert addr0 == addr1, f"Expected same address on different cores (per-bank allocators), got {addr0} vs {addr1}"


def test_per_core_round_trip(device):
    """Data written to a per-core allocated tensor reads back correctly."""
    shard_bytes = 2048
    core = ttnn.CoreCoord(0, 0)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        [1, shard_bytes],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
        per_core_shard_sizes=[shard_bytes],
    )
    data = torch.arange(shard_bytes, dtype=torch.uint8).reshape(1, shard_bytes)
    tensor = ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    result = ttnn.to_torch(ttnn.from_device(tensor))
    assert torch.equal(data, result), "Round-trip data mismatch"


def test_per_core_tetris_allocation(device):
    """Tetris-style allocation across 4 cores with alloc/free/realloc patterns.

    Exercises independent per-bank free-lists with varied sizes, frees, and
    reallocations. All expected addresses are computed by PerCoreMemMap and
    validated against actual device addresses.
    """
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]

    # Allocation script: (label, action, core_index, size)
    #   action: "alloc" or "free"
    #   For "free", size is ignored — we free the tensor stored under label.
    script = [
        # Round 1: initial allocations across all cores
        ("c0_a", "alloc", 0, 2048),
        ("c1_a", "alloc", 1, 4096),
        ("c2_a", "alloc", 2, 1024),
        ("c3_a", "alloc", 3, 3072),
        # Round 2: second alloc on some cores
        ("c0_b", "alloc", 0, 1024),
        ("c1_b", "alloc", 1, 512),
        ("c2_b", "alloc", 2, 2048),
        # Round 3: free some, then reallocate
        ("c0_a", "free", 0, 0),
        ("c1_a", "free", 1, 0),
        ("c0_c", "alloc", 0, 2048),  # reuses c0_a's freed space
        ("c1_c", "alloc", 1, 2048),  # fits in c1_a's freed 4096 gap
        ("c1_d", "alloc", 1, 1024),  # also fits in remaining c1_a gap
        # Round 4: more on core 3
        ("c3_b", "alloc", 3, 512),
        ("c3_a", "free", 3, 0),
        ("c3_c", "alloc", 3, 3072),  # reuses c3_a's freed space
    ]

    tensors = {}  # label → ttnn.Tensor
    actual = {}
    mem = None

    for label, action, core_idx, size in script:
        if action == "alloc":
            t = _create_single_core_tensor(device, cores[core_idx], size)
            if mem is None:
                mem = PerCoreMemMap(t.buffer_address(), size)
            addr = mem.alloc(label, core_idx, size)
            actual[label] = t.buffer_address()
            tensors[label] = t
        elif action == "free":
            freed_addr = mem.expected[label]
            del tensors[label]
            mem.free(core_idx, freed_addr)
            del actual[label]
            del mem.expected[label]

    mem.validate(actual)


def _create_lockstep_tensor(device, core, shard_bytes):
    """Create a single-core HEIGHT_SHARDED L1 tensor with lockstep (default) allocation."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        [1, shard_bytes],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    data = torch.zeros(1, shard_bytes, dtype=torch.uint8)
    return ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )


def _addr_ranges_overlap(addr_a, size_a, addr_b, size_b):
    """Check if two address ranges [a, a+size_a) and [b, b+size_b) overlap."""
    return addr_a < addr_b + size_b and addr_b < addr_a + size_a


def test_per_core_and_lockstep_coexist(device):
    """Interleave per-core and lockstep allocations across multiple cores.

    Verifies:
    - Per-core and lockstep use different allocator pools (no address reuse)
    - No address range overlaps between any per-core and lockstep allocation
    - Freeing per-core doesn't affect lockstep, and vice versa
    - Both pools can allocate independently after the other has allocated
    """
    num_cores = 4
    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]

    # Track all allocations: (label, addr, size, kind)
    allocs = []
    tensors = {}

    # Round 1: per-core allocations on each core (different sizes)
    pc_sizes = [2048, 4096, 1024, 3072]
    for i, size in enumerate(pc_sizes):
        label = f"pc_c{i}_{size}"
        t = _create_single_core_tensor(device, cores[i], size)
        tensors[label] = t
        allocs.append((label, t.buffer_address(), size, "per_core"))

    # Round 2: lockstep allocations on same cores
    ls_sizes = [1024, 2048, 512, 1024]
    for i, size in enumerate(ls_sizes):
        label = f"ls_c{i}_{size}"
        t = _create_lockstep_tensor(device, cores[i], size)
        tensors[label] = t
        allocs.append((label, t.buffer_address(), size, "lockstep"))

    # Round 3: more per-core after lockstep
    for i, size in enumerate([512, 1024]):
        label = f"pc2_c{i}_{size}"
        t = _create_single_core_tensor(device, cores[i], size)
        tensors[label] = t
        allocs.append((label, t.buffer_address(), size, "per_core"))

    # Round 4: free some per-core, allocate lockstep in the freed cores
    del tensors["pc_c0_2048"]
    del tensors["pc_c1_4096"]
    ls_after_free_sizes = [2048, 2048]
    for i, size in enumerate(ls_after_free_sizes):
        label = f"ls_after_free_c{i}_{size}"
        t = _create_lockstep_tensor(device, cores[i], size)
        tensors[label] = t
        allocs.append((label, t.buffer_address(), size, "lockstep"))

    # Round 5: free some lockstep, allocate per-core
    del tensors["ls_c2_512"]
    del tensors["ls_c3_1024"]
    pc_after_free_sizes = [512, 1024]
    for i, size in enumerate(pc_after_free_sizes):
        label = f"pc_after_free_c{i+2}_{size}"
        t = _create_single_core_tensor(device, cores[i + 2], size)
        tensors[label] = t
        allocs.append((label, t.buffer_address(), size, "per_core"))

    # Validate: no per-core allocation overlaps with any lockstep allocation on the same core
    # (Per-core allocs on different cores CAN share addresses — that's the point)
    # Group by core for overlap checking
    for i in range(num_cores):
        core_pc = [(l, a, s) for l, a, s, k in allocs if k == "per_core" and l in tensors and f"_c{i}_" in l]
        core_ls = [(l, a, s) for l, a, s, k in allocs if k == "lockstep" and l in tensors and f"_c{i}_" in l]
        for pc_label, pc_addr, pc_size in core_pc:
            for ls_label, ls_addr, ls_size in core_ls:
                assert not _addr_ranges_overlap(pc_addr, pc_size, ls_addr, ls_size), (
                    f"Overlap on core {i}: {pc_label}=[{pc_addr:#x}, +{pc_size}) vs "
                    f"{ls_label}=[{ls_addr:#x}, +{ls_size})"
                )

    # Validate: all remaining tensors are still allocated
    for label, t in tensors.items():
        assert t.is_allocated(), f"{label} should still be allocated"
