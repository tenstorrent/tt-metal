# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for per-core L1 allocation via MemoryConfig.experimental_set_per_core_allocation().

Each test creates single-core HEIGHT_SHARDED tensors with experimental_set_per_core_allocation(),
which triggers the per-bank allocator (AllocatorID bank_id+1) instead of lockstep.

The local conftest.py sets TT_METAL_ALLOCATOR_MODE_HYBRID=1 and creates the device.
"""

import torch
import ttnn


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
            f"  {k}: expected={self.expected[k]:#x} actual={'MISSING' if k not in actual else f'{actual[k]:#x}'}"
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
    )
    mem_config.experimental_set_per_core_allocation(True)
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
    addr0 = t0.experimental_per_core_buffer_address(core0)
    addr1 = t1.experimental_per_core_buffer_address(core1)
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
    )
    mem_config.experimental_set_per_core_allocation(True)
    data = torch.arange(shard_bytes, dtype=torch.uint8).reshape(1, shard_bytes)
    tensor = ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    result = ttnn.to_torch(ttnn.from_device(tensor))
    assert torch.equal(data, result), "Round-trip data mismatch"


def test_per_core_sharded_dealloc_realloc(device):
    """Deallocate a per-core sharded tensor and reallocate — verifies deallocation frees per-core space.

    1. Allocate a per-core sharded tensor across all cores
    2. Record per-core addresses
    3. Delete it
    4. Allocate again — should get the same addresses (space was freed and reused)
    """
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            cores.append(ttnn.CoreCoord(x, y))

    SHARD_BYTES = 2048
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    shard_spec = ttnn.ShardSpec(core_grid, [1, SHARD_BYTES], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    mem_config.experimental_set_per_core_allocation(True)
    data = torch.zeros(num_cores, SHARD_BYTES, dtype=torch.uint8)

    # First allocation
    t1 = ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config)
    addrs1 = [t1.experimental_per_core_buffer_address(c) for c in cores]

    # Deallocate
    del t1

    # Second allocation — should reuse the same per-core space
    t2 = ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config)
    addrs2 = [t2.experimental_per_core_buffer_address(c) for c in cores]

    assert (
        addrs1 == addrs2
    ), f"Expected same addresses after dealloc/realloc:\n  first:  {[f'{a:#x}' for a in addrs1]}\n  second: {[f'{a:#x}' for a in addrs2]}"


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
                mem = PerCoreMemMap(t.experimental_per_core_buffer_address(cores[core_idx]), size)
            mem.alloc(label, core_idx, size)
            actual[label] = t.experimental_per_core_buffer_address(cores[core_idx])
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
        allocs.append((label, t.experimental_per_core_buffer_address(cores[i]), size, "per_core"))

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
        allocs.append((label, t.experimental_per_core_buffer_address(cores[i]), size, "per_core"))

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
        allocs.append((label, t.experimental_per_core_buffer_address(cores[i + 2]), size, "per_core"))

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


def test_all_cores_lockstep_then_per_core_then_reverse(device):
    """Allocate on ALL L1 cores: lockstep first then per-core, then deallocate and reverse order.

    This stresses the dependency-aware path in the bank manager:
    - Phase 1: lockstep on all cores → per-core on all cores (lockstep first, deps empty initially)
    - Phase 2: deallocate all → per-core on all cores → lockstep on all cores (deps non-empty for lockstep)
    Validates no overlaps in either phase.
    """
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            cores.append(ttnn.CoreCoord(x, y))

    shard_bytes = 2048

    # --- Phase 1: lockstep first, then per-core ---
    lockstep_tensors = {}
    per_core_tensors = {}

    # Lockstep on all cores
    for i, core in enumerate(cores):
        lockstep_tensors[i] = _create_lockstep_tensor(device, core, shard_bytes)

    # Per-core on all cores
    for i, core in enumerate(cores):
        per_core_tensors[i] = _create_single_core_tensor(device, core, shard_bytes)

    # Validate: no overlaps between lockstep and per-core on same core
    for i in range(num_cores):
        ls_addr = lockstep_tensors[i].buffer_address()
        pc_addr = per_core_tensors[i].experimental_per_core_buffer_address(cores[i])
        assert not _addr_ranges_overlap(
            ls_addr, shard_bytes, pc_addr, shard_bytes
        ), f"Phase 1 overlap on core {i}: lockstep={ls_addr:#x} per_core={pc_addr:#x}"

    # Deallocate all
    lockstep_tensors.clear()
    per_core_tensors.clear()

    # --- Phase 2: per-core first, then lockstep ---

    # Per-core on all cores (varying sizes)
    pc_sizes = [1024 + (i % 4) * 1024 for i in range(num_cores)]  # 1024, 2048, 3072, 4096, ...
    for i, core in enumerate(cores):
        per_core_tensors[i] = _create_single_core_tensor(device, core, pc_sizes[i])

    # Lockstep on all cores — this triggers the dependency-aware path
    # because per-bank allocators now have allocations
    for i, core in enumerate(cores):
        lockstep_tensors[i] = _create_lockstep_tensor(device, core, shard_bytes)

    # Validate: no overlaps
    for i in range(num_cores):
        ls_addr = lockstep_tensors[i].buffer_address()
        pc_addr = per_core_tensors[i].experimental_per_core_buffer_address(cores[i])
        assert not _addr_ranges_overlap(ls_addr, shard_bytes, pc_addr, pc_sizes[i]), (
            f"Phase 2 overlap on core {i}: lockstep={ls_addr:#x}+{shard_bytes} " f"per_core={pc_addr:#x}+{pc_sizes[i]}"
        )

    # All tensors still alive
    for i in range(num_cores):
        assert lockstep_tensors[i].is_allocated()
        assert per_core_tensors[i].is_allocated()


def test_per_core_width_sharded_tiled_bfp4_round_trip(device):
    """FORMAT coverage: per-core allocation coexists with WIDTH_SHARDED + TILE_LAYOUT +
    BFLOAT4_B for a (7168, 576) tensor on the 18-core grid (0,8)-(8,9), with a couple of
    co-resident per-core tensors on (0,8)/(0,9) (so per-core addresses are non-uniform).

    NOTE: this is a host ``from_torch``/``to_torch`` round-trip, which is SYMMETRIC (both
    sides used the same buffer address), so it does NOT catch the per-core data-movement
    address bug — it passed even while that bug was live, because the isolated co-resident
    topology differs from the real dense run. The authoritative guard for that bug is
    ``test_per_core_kernel_readback_honors_per_core_address`` (kernel reads at the per-core
    address). This test only verifies that the BFP4/TILE/WIDTH-sharded per-core creation
    path round-trips at all.
    """
    H, W = 7168, 576
    kv_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9))])
    kv_cores = list(ttnn.corerange_to_cores(kv_grid, row_wise=True))
    num_cores = len(kv_cores)
    assert num_cores == 18
    shard_w = W // num_cores  # 32

    # Phase 1: prior per-core alloc on (0,8) and (0,9) — the two cores that
    # relocate in the real run (kv_norm gamma / k_rope co-residents).
    relocate_cores = [ttnn.CoreCoord(0, 8), ttnn.CoreCoord(0, 9)]
    prealloc = [_create_single_core_tensor(device, c, 2048) for c in relocate_cores]

    # Phase 2: WIDTH_SHARDED, TILE_LAYOUT, BFP4 per-core tensor with distinct
    # nonzero data per column-shard (column block c filled with value c+1).
    data = torch.zeros(H, W, dtype=torch.float32)
    for c in range(num_cores):
        data[:, c * shard_w : (c + 1) * shard_w] = float(c + 1)

    shard_spec = ttnn.ShardSpec(kv_grid, [H, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    mem_config.experimental_set_per_core_allocation(True)

    tensor = ttnn.from_torch(
        data, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    addrs = {(c.x, c.y): tensor.experimental_per_core_buffer_address(c) for c in kv_cores}
    base = addrs[(1, 8)]
    relocated = [(c.x, c.y) for c in kv_cores if addrs[(c.x, c.y)] != base]
    print(f"\n[REPRO-KVA] distinct_addrs={len(set(addrs.values()))} relocated={relocated} base_addr={base}")

    result = ttnn.to_torch(ttnn.from_device(tensor)).float()
    # per column-shard: mean abs (BFP4 is lossy, but zero stays ~zero)
    zero_shards = []
    for c in range(num_cores):
        block = result[:, c * shard_w : (c + 1) * shard_w]
        if float(block.abs().mean().item()) < 1e-3:
            zero_shards.append((c, (kv_cores[c].x, kv_cores[c].y)))
    print(f"[REPRO-KVA] zero_shards={len(zero_shards)}/{num_cores}: {zero_shards}")
    assert not zero_shards, (
        f"{len(zero_shards)}/{num_cores} WIDTH-sharded BFP4 per-core column-shards read back ZERO "
        f"(BFP4/TILE/WIDTH per-core creation round-trip is broken): {zero_shards}"
    )


# Inline data-movement kernel: copy `num_bytes` from a per-core SOURCE L1 address
# (passed as a runtime arg, exactly how blaze wires weights via
# experimental_per_core_buffer_address(core)) into a lockstep DESTINATION L1 address
# on the same core. A plain local L1->L1 word copy — no CB / NOC / TensorAccessor needed.
_PER_CORE_READBACK_KERNEL = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr  = get_arg_val<uint32_t>(0);  // per-core source base (kernel's view)
    const uint32_t dst_addr  = get_arg_val<uint32_t>(1);  // lockstep destination base
    const uint32_t num_bytes = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const uint32_t num_words = num_bytes >> 2;
    for (uint32_t i = 0; i < num_words; ++i) {
        dst[i] = src[i];
    }
}
"""


def test_per_core_kernel_readback_honors_per_core_address(device):
    """REGRESSION test for the per-core data-movement address bug.

    Unlike the host round-trip tests above, this introduces the asymmetry the bug
    needs: a kernel reads each core's shard at the per-core address
    (``experimental_per_core_buffer_address(core)``, wired as a runtime arg exactly how
    blaze's matmul reads weights), while ``from_torch`` writes via the host
    data-movement path.

    The defect: host data movement wrote every core's shard at the uniform scalar
    ``buffer.address()`` (== cores[0]'s address) plus the core's logical page offset,
    instead of at each core's independent per-core address. Host round-trips
    (``to_torch``) can't see this because they use the same scalar on both sides. A
    kernel reading at the per-core address CAN: a core whose per-core address sits
    *below* the scalar by more than the write's page span reads whatever is there
    (zeros), not the written data — exactly what made 16/18 kv_a cores read zero
    (-> kv-cache inf) in the full dense run.

    To reproduce robustly in isolation we recreate that key geometry directly: a
    per-core triangle pre-alloc with a step (``PREALLOC_STEP``) far LARGER than the
    shard/page size pushes each core's per-core address far below cores[0]'s (the
    scalar), while the buggy host write only advances by the small page size. So the
    written bytes and the kernel's read address diverge on (almost) every core.

    With the fix (host data movement honors ``get_per_core_address``), the kernel reads
    real data on every core. With the fix reverted, the diverged cores read zeros and
    this test FAILS.
    """
    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    num_cores = len(cores)

    SHARD_BYTES = 1024  # small page; word-aligned
    PREALLOC_STEP = 4096  # > SHARD_BYTES: spreads per-core addresses apart faster than
    #                       the host write advances (and stays within an L1 bank: the
    #                       largest core's pre-alloc is num_cores * PREALLOC_STEP)

    # Phase 1: triangle pre-alloc with a LARGE step. cores[0] consumes the least (=>
    # highest address => the scalar buffer.address()); each later core sits a full
    # PREALLOC_STEP lower. The buggy host write advances only by SHARD_BYTES per core,
    # so it never reaches the far-lower per-core addresses.
    triangle_tensors = {}
    for i, core in enumerate(cores):
        triangle_tensors[i] = _create_single_core_tensor(device, core, (i + 1) * PREALLOC_STEP)

    # Phase 2: the SOURCE per-core sharded tensor with DISTINCT NONZERO data per core.
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    shard_spec = ttnn.ShardSpec(core_grid, [1, SHARD_BYTES], ttnn.ShardOrientation.ROW_MAJOR)
    src_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    src_mem.experimental_set_per_core_allocation(True)

    data = torch.zeros(num_cores, SHARD_BYTES, dtype=torch.uint8)
    for i in range(num_cores):
        data[i, :] = (i % 255) + 1
    src = ttnn.from_torch(data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=src_mem)

    # Precondition: per-core addresses must be non-uniform, else scalar == per-core and
    # the bug is structurally invisible.
    src_addrs = [src.experimental_per_core_buffer_address(c) for c in cores]
    assert len(set(src_addrs)) > 1, f"expected non-uniform per-core addresses; got {len(set(src_addrs))} distinct"

    # DESTINATION: a lockstep (uniform-address) uint8 tensor the kernel copies into,
    # read back reliably via to_torch.
    dst_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    dst = ttnn.from_torch(
        torch.zeros(num_cores, SHARD_BYTES, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dst_mem,
    )
    dst_base = dst.buffer_address()  # lockstep: uniform across cores

    # Wire the kernel: per core, source = the per-core address (the kernel's true view),
    # destination = the lockstep base.
    rt_args = ttnn.RuntimeArgs()
    for c in cores:
        rt_args[c.x][c.y] = [src.experimental_per_core_buffer_address(c), dst_base, SHARD_BYTES]

    kernel = ttnn.KernelDescriptor(
        kernel_source=_PER_CORE_READBACK_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=[],
        runtime_args=rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])

    ttnn.generic_op([src, dst], program)

    result = ttnn.to_torch(ttnn.from_device(dst))
    mismatched = [i for i in range(num_cores) if not torch.equal(result[i], data[i])]
    zero_rows = [i for i in range(num_cores) if int(result[i].max().item()) == 0]
    print(
        f"\n[KERNEL-READBACK] distinct_src_addrs={len(set(src_addrs))} "
        f"mismatched={len(mismatched)}/{num_cores} zero={len(zero_rows)}/{num_cores}"
    )
    assert not mismatched, (
        f"{len(mismatched)}/{num_cores} cores' kernel-read shards differ from the written data "
        f"({len(zero_rows)} read ALL-ZERO) — host data movement wrote at the uniform scalar "
        f"buffer.address() instead of the per-core address the kernel reads "
        f"(reproduces dense kv_a inf). rows={mismatched[:32]}"
    )

    for i in range(num_cores):
        assert triangle_tensors[i].is_allocated()


def test_triangle_allocation_then_uniform_sharded(device):
    """Triangle per-core allocation on ALL compute cores, then a per-core sharded tensor.

    Phase 1: Create increasing-size per-core tensors (triangle pattern) on every
    core in the compute grid. Core at flat index i gets (i+1) * TILE_BYTES.
    This consumes different amounts of L1 per core.

    Phase 2: Allocate one HEIGHT_SHARDED per-core tensor across all cores.
    Since each core's per-bank allocator has consumed a different amount
    (the triangle), the resulting per-core addresses should be strictly
    decreasing with flat index (inverse triangle).
    """
    grid = device.compute_with_storage_grid_size()
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            cores.append(ttnn.CoreCoord(x, y))
    num_cores = len(cores)

    TILE_BYTES = 1024  # above DRAM alignment to avoid min_allocation_size issues

    # Phase 1: Triangle allocation — flat index i gets (i+1) * TILE_BYTES
    triangle_tensors = {}
    for i, core in enumerate(cores):
        size = (i + 1) * TILE_BYTES
        triangle_tensors[i] = _create_single_core_tensor(device, core, size)

    # Phase 2: Allocate one HEIGHT_SHARDED per-core tensor across all cores
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    shard_bytes = TILE_BYTES
    shard_spec = ttnn.ShardSpec(core_grid, [1, shard_bytes], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    mem_config.experimental_set_per_core_allocation(True)
    sharded_data = torch.zeros(num_cores, shard_bytes, dtype=torch.uint8)
    sharded_tensor = ttnn.from_torch(
        sharded_data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )

    # Verify: per-core addresses form an inverse triangle
    # Core at flat index 0 consumed least → highest address
    # Core at flat index N-1 consumed most → lowest address
    addrs = [sharded_tensor.experimental_per_core_buffer_address(c) for c in cores]
    for i in range(num_cores - 1):
        assert addrs[i] > addrs[i + 1], (
            f"Expected inverse triangle: core {i} ({cores[i].x},{cores[i].y}) addr={addrs[i]:#x} "
            f"should be > core {i+1} ({cores[i+1].x},{cores[i+1].y}) addr={addrs[i+1]:#x}"
        )

    # All tensors still alive
    for i in range(num_cores):
        assert triangle_tensors[i].is_allocated()
    assert sharded_tensor.is_allocated()
