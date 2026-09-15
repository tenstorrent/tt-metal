# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real allocation scenarios: concurrent requests, full-pool reuse, and large contexts."""

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_slow_dispatch

PAGE_SIZE = 32


class CachePool:
    """Create metadata, submit token ranges, and check counts and ownership on every replica."""

    def __init__(self, device, *, slots=3, context=288, sp=2, bundles_per_bank=3, chunk_size=None):
        self.sp, self.bundles_per_bank = sp, bundles_per_bank
        self.banks = device.dram_grid_size().x
        pools = sp * self.banks
        self.chunk_size = chunk_size if chunk_size is not None else PAGE_SIZE * sp
        pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
        chunk_pages = self.chunk_size // PAGE_SIZE
        local_chunk_pages = chunk_pages // sp
        logical = torch.arange(pages, dtype=torch.int32)
        self.owner = (logical % chunk_pages) // local_chunk_pages
        self.local_page = (logical // chunk_pages) * local_chunk_pages + logical % local_chunk_pages
        self.pool_pages = [
            torch.where((self.owner == owner_sp) & (self.local_page % self.banks == bank))[0]
            for bank in range(self.banks)
            for owner_sp in range(sp)
        ]
        self.tensors = [
            ttnn.from_torch(
                value,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            for value in (
                torch.zeros((slots, pages), dtype=torch.int32),
                torch.zeros((1, slots), dtype=torch.int32),
                torch.arange(bundles_per_bank - 1, -1, -1, dtype=torch.int32).repeat(pools, 1),
                torch.full((1, pools), bundles_per_bank, dtype=torch.int32),
            )
        ]

    def update(self, slot, start, end):
        return ttnn.experimental.update_cache_bundle_allocation(
            *self.tensors,
            slot_id=slot,
            actual_start=start,
            actual_end=end,
            chunk_size=self.chunk_size,
            page_size=PAGE_SIZE,
        )

    def check(self, pages_per_slot):
        state = []
        for name, tensor in zip(("page_table", "allocated_pages", "free_list", "free_count"), self.tensors):
            replicas = [ttnn.to_torch(shard).to(torch.int32) for shard in ttnn.get_device_tensors(tensor)]
            assert all(torch.equal(replica, replicas[0]) for replica in replicas), f"Replicas disagree on {name}"
            state.append(replicas[0])
        table, allocated, free, counts = state
        assert allocated[0].tolist() == pages_per_slot
        for bank in range(self.banks):
            for sp in range(self.sp):
                row = bank * self.sp + sp
                indices = self.pool_pages[row]
                live = torch.cat([table[slot, indices[indices < pages]] for slot, pages in enumerate(pages_per_slot)])
                assert torch.all(live % self.banks == bank), f"Wrong bank for SP{sp}, bank{bank}"
                available = int(counts[0, row])
                assert available == self.bundles_per_bank - len(live), f"Wrong count for SP{sp}, bank{bank}"
                owners = torch.cat((live, free[row, :available] * self.banks + bank)).sort().values
                expected = torch.arange(self.bundles_per_bank, dtype=torch.int32) * self.banks + bank
                assert torch.equal(owners, expected), f"Invalid ownership for SP{sp}, bank{bank}"
        return state


def test_concurrent_requests_and_slot_reuse(device):
    device.enable_program_cache()
    pools = [CachePool(device) for _ in range(2)]  # Distinct live buffers exercise program-cache reuse.
    for pool_index, pool in enumerate(pools):
        banks = pool.banks
        # slot, token start/end, expected live bundle IDs for each slot.
        requests = [
            (0, 0, 33, [[0, 0], [], []]),  # Prefill.
            (0, 33, 64, [[0, 0], [], []]),  # Decode within allocated pages.
            (0, 64, 65, [[0, 0, 1], [], []]),  # Decode across a page boundary.
            (0, 64, 65, [[0, 0, 1], [], []]),  # Repeated growth is idempotent.
            (1, 0, 33, [[0, 0, 1], [banks, banks], []]),  # A concurrent request.
            (0, 0, 0, [[], [banks, banks], []]),  # Completion returns bundles to the pool.
            (2, 0, 65, [[], [banks, banks], [0, 0, 1]]),  # Another request reuses them.
            (2, 0, 33, [[], [banks, banks], [0, 0]]),  # Replace a slot with a shorter prompt.
            (1, 0, 0, [[], [], [0, 0]]),
            (2, 0, 0, [[], [], []]),
            (2, 0, 0, [[], [], []]),  # Repeated release must not duplicate free IDs.
        ]
        for slot, start, end, expected in requests:
            pool.update(slot, start, end)
            table, _, _, _ = pool.check([len(ids) for ids in expected])
            for row, ids in zip(table, expected):
                assert row[: len(ids)].tolist() == ids
        if pool_index == 0:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries


@pytest.mark.parametrize("sp", [1, 3, 8])
def test_bank_affinity_across_request_continuations(device, sp):
    pool = CachePool(device, slots=3, context=10000, sp=sp, bundles_per_bank=64)
    pages_5k = (5000 + PAGE_SIZE - 1) // PAGE_SIZE
    pages_6k = (6000 + PAGE_SIZE - 1) // PAGE_SIZE
    pool.update(0, 0, 5000)
    first = pool.check([pages_5k, 0, 0])[0].clone()
    assert torch.equal(first[0, :pages_5k], torch.arange(pages_5k, dtype=torch.int32) // sp)

    pool.update(1, 0, 5000)
    before = pool.check([pages_5k, pages_5k, 0])[0].clone()
    assert torch.equal(before[0], first[0])
    # Another request's logical page zero must use bank zero as well.
    assert before[1, 0] % pool.banks == 0
    assert before[1, 0] != before[0, 0]
    if sp == 1 and pool.banks == 8:
        assert before[1, 0] == 160  # B0..B156 belong to the first 5k request.

    snapshot = pool.check([pages_5k, pages_5k, 0])
    pool.update(0, 5000, 5010)  # Fill more of the same partial page; no new allocation.
    assert all(torch.equal(a, b) for a, b in zip(snapshot, pool.check([pages_5k, pages_5k, 0])))
    pool.update(0, 5010, 6000)
    grown = pool.check([pages_6k, pages_5k, 0])[0].clone()
    assert torch.equal(grown[0, :pages_5k], before[0, :pages_5k])
    assert torch.equal(grown[1], before[1])
    snapshot = pool.check([pages_6k, pages_5k, 0])
    pool.update(0, 5010, 6000)  # Repeating the continuation cannot consume more IDs.
    assert all(torch.equal(a, b) for a, b in zip(snapshot, pool.check([pages_6k, pages_5k, 0])))

    pool.update(0, 0, 0)
    pool.check([0, pages_5k, 0])
    pool.update(2, 0, 6000)
    reused = pool.check([0, pages_5k, pages_6k])[0]
    assert torch.equal(reused[1], grown[1])
    for owner_sp in range(sp):
        assert torch.equal(
            reused[2, owner_sp:pages_6k:sp].sort().values,
            grown[0, owner_sp:pages_6k:sp].sort().values,
        )


def test_full_pool_release_and_reuse(device, expect_error):
    sp = 2
    pages = sp * device.dram_grid_size().x
    tokens = pages * PAGE_SIZE
    pool = CachePool(device, context=tokens, sp=sp, bundles_per_bank=1)
    result = pool.update(0, 0, tokens)
    assert result.buffer_address() == pool.tensors[0].buffer_address()
    before = pool.check([pages, 0, 0])
    pool.update(0, 0, tokens)  # Every bank is full; reset must reuse this slot's own bundles.
    pool.check([pages, 0, 0])
    with expect_error(RuntimeError, "actual_start"):
        pool.update(1, 64, 32)
    assert all(torch.equal(old, new) for old, new in zip(before, pool.check([pages, 0, 0])))
    pool.update(0, 0, 0)
    pool.check([0, 0, 0])
    pool.update(1, 0, tokens)
    table, _, _, _ = pool.check([0, pages, 0])
    assert table[1, :pages].tolist() == torch.arange(pool.banks).repeat_interleave(sp).tolist()


@pytest.mark.parametrize(
    "slots,context,sp",
    [(500, 1_048_576, 8), (500, 1_048_576, 32), (1, 65_537 * PAGE_SIZE, 1)],
    ids=["500_users_1m_context_sp8", "500_users_1m_context_sp32", "single_context_over_65535_pages"],
)
def test_large_contexts(device, slots, context, sp):
    pages = context // PAGE_SIZE
    banks = device.dram_grid_size().x
    local_pages = pages // sp
    bundles_per_bank = slots * ((local_pages + banks - 1) // banks)
    pool = CachePool(device, slots=slots, context=context, sp=sp, bundles_per_bank=bundles_per_bank)
    for slot in range(slots):
        pool.update(slot, 0, context)
    table, _, _, _ = pool.check([pages] * slots)
    # Per-bank consumption can differ when the local page count is not a multiple of banks.
    local = torch.arange(local_pages, dtype=torch.int32)
    per_bank = local_pages // banks + (local % banks < local_pages % banks).to(torch.int32)
    expected = (local[None, :] + torch.arange(slots)[:, None] * per_bank[None, :] * banks).repeat_interleave(sp, dim=1)
    assert torch.equal(table, expected)

    # Replace a full context using its own bundles, preserving every other slot.
    slot = slots // 2
    pool.update(slot, 0, context)
    table, _, _, _ = pool.check([pages] * slots)
    # LIFO reuse reverses each bank's IDs independently, preserving bank affinity.
    for bank in range(banks):
        for owner_sp in range(sp):
            row = bank * sp + owner_sp
            expected[slot, row :: sp * banks] = expected[slot, row :: sp * banks].flip(0)
    assert torch.equal(table, expected)

    for slot in range(slots):
        pool.update(slot, 0, 0)
    pool.check([0] * slots)  # Every bundle is available again, including IDs above UINT16.


@pytest.mark.parametrize("sp", [1, 8, 17, 32, 128])
def test_chunked_growth_preserves_other_slots(device, sp):
    # Large table rows, unaligned slot counters and free-list tails, and 4 KiB crossings.
    banks = device.dram_grid_size().x
    max_pages = 1600 * banks
    pool = CachePool(device, slots=19, context=max_pages * PAGE_SIZE, sp=sp, bundles_per_bank=2051)
    pages = [0] * 19
    for slot in (0, 16, 18):
        pool.update(slot, 0, 33)
        pages[slot] = 2
    before = pool.check(pages)[0].clone()
    for end_pages in sorted({160, 161, sp * banks - 1, sp * banks + 1, 1023 * banks, 1025 * banks, max_pages}):
        old_pages = pages[17]
        pool.update(17, old_pages * PAGE_SIZE, end_pages * PAGE_SIZE)
        pages[17] = end_pages
        table, _, _, _ = pool.check(pages)
        assert torch.equal(table[:17], before[:17])
        assert torch.equal(table[18:], before[18:])
        assert torch.equal(table[17, :old_pages], before[17, :old_pages])
        assert torch.equal(table[17, end_pages:], before[17, end_pages:])
        before = table.clone()
    snapshot = pool.check(pages)
    pool.update(17, 1, 2)  # Already covered: no metadata changes.
    assert all(torch.equal(old, new) for old, new in zip(snapshot, pool.check(pages)))
    for end_pages in (17, 1201, 0):  # Reset shorter, reset larger, then release.
        pool.update(17, 0, end_pages * PAGE_SIZE)
        pages[17] = end_pages
        table, _, _, _ = pool.check(pages)
        assert torch.equal(table[:17], before[:17])
        assert torch.equal(table[18:], before[18:])


def request_tensor(value, device):
    return ttnn.from_torch(
        torch.tensor([[value]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def test_tensor_requests(device):
    device.enable_program_cache()
    # Retain both sets of buffers so the second call must override cached addresses.
    pools = [CachePool(device, bundles_per_bank=8) for _ in range(2)]
    request_buffers = []
    entries = None
    for pool in pools:
        reference = CachePool(device, bundles_per_bank=8)
        for slot, start, end, pages in (
            (0, 0, 65, [3, 0, 0]),
            (1, 0, 33, [3, 2, 0]),
            (0, 65, 129, [5, 2, 0]),
            (0, 100, 128, [5, 2, 0]),
            (0, 0, 32, [1, 2, 0]),
            (1, 0, 0, [1, 0, 0]),
        ):
            args = [request_tensor(value, device) for value in (slot, start, end)]
            request_buffers.append(args)
            pool.update(*args)
            reference.update(slot, start, end)
            assert all(torch.equal(a, b) for a, b in zip(pool.check(pages), reference.check(pages)))
            if entries is None:
                entries = device.num_program_cache_entries()
            assert device.num_program_cache_entries() == entries


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
@skip_for_slow_dispatch()
def test_trace_replay_uses_updated_request_values(device):
    device.enable_program_cache()
    pool = CachePool(device, context=10240, sp=8, bundles_per_bank=128, chunk_size=5120)
    args = [request_tensor(0, device) for _ in range(3)]
    pool.update(*args)  # Compile before capture with an empty-slot release.
    entries = device.num_program_cache_entries()
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    result = pool.update(*args)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        # slot, start, end, expected allocated pages per slot.
        for slot, start, end, pages in (
            (0, 0, 5120, [160, 0, 0]),  # First chunk.
            (0, 5120, 10240, [320, 0, 0]),  # Grow by one chunk.
            (0, 5120, 10240, [320, 0, 0]),  # Repeated growth is a no-op.
            (1, 0, 5120, [320, 160, 0]),  # Switch slots in the same trace.
            (0, 0, 32, [1, 160, 0]),  # Reset to a shorter request.
            (1, 0, 0, [1, 0, 0]),  # Release.
            (0, 0, 0, [0, 0, 0]),
            (0, 0, 0, [0, 0, 0]),  # Repeated release.
            (2, 0, 5120, [0, 0, 160]),  # Reuse returned bundles.
        ):
            for tensor, value in zip(args, (slot, start, end)):
                host = ttnn.from_torch(
                    torch.tensor([[value]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                ttnn.copy_host_to_device_tensor(host, tensor, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            pool.check(pages)
            assert result.buffer_address() == pool.tensors[0].buffer_address()
            assert device.num_program_cache_entries() == entries
    finally:
        ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "placement"])
def test_invalid_request_tensor(device, expect_error, invalid):
    pool = CachePool(device)
    value = ttnn.from_torch(
        torch.zeros((1, 2) if invalid == "shape" else (1, 1), dtype=torch.int32),
        dtype=ttnn.int32 if invalid == "dtype" else ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG if invalid == "placement" else ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    with expect_error(RuntimeError, "Request must"):
        pool.update(request_tensor(0, device), request_tensor(0, device), value)
    pool.check([0, 0, 0])


@pytest.mark.parametrize("tensor_mask", range(1, 7))
@pytest.mark.parametrize("warm_cache", [False, True])
def test_mixed_request_inputs_rejected(device, expect_error, tensor_mask, warm_cache):
    device.enable_program_cache()
    pool = CachePool(device)
    if warm_cache:
        pool.update(0, 0, 32)
        pool.update(*(request_tensor(value, device) for value in (0, 0, 32)))
    pages = [1, 0, 0] if warm_cache else [0, 0, 0]
    before = pool.check(pages)
    args = [request_tensor(value, device) if tensor_mask & (1 << i) else value for i, value in enumerate((0, 0, 64))]
    entries = device.num_program_cache_entries()
    with expect_error(TypeError, "incompatible function arguments"):
        pool.update(*args)
    assert device.num_program_cache_entries() == entries
    assert all(torch.equal(old, new) for old, new in zip(before, pool.check(pages)))


@pytest.mark.parametrize("invalid", ["pool_rows", "count_shape"])
def test_invalid_bank_pool_shapes(device, expect_error, invalid):
    pool = CachePool(device, sp=1)
    rows = pool.banks + 1

    def metadata(value):
        return ttnn.from_torch(
            value,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    tensors = pool.tensors.copy()
    tensors[3] = metadata(torch.full((1, rows), pool.bundles_per_bank, dtype=torch.int32))
    if invalid == "pool_rows":
        tensors[2] = metadata(torch.zeros((rows, pool.bundles_per_bank), dtype=torch.int32))
    message = "free_list rows" if invalid == "pool_rows" else "free_count must have shape"
    with expect_error(RuntimeError, message):
        ttnn.experimental.update_cache_bundle_allocation(
            *tensors, slot_id=0, actual_start=0, actual_end=32, chunk_size=pool.chunk_size, page_size=PAGE_SIZE
        )
    pool.check([0, 0, 0])


@pytest.mark.parametrize("sp,local_chunk_pages", [(1, 20), (3, 5), (8, 20), (8, 1), (8, 32)])
def test_chunk_distribution_and_continuation(device, sp, local_chunk_pages):
    chunk_pages = sp * local_chunk_pages
    chunk_size = chunk_pages * PAGE_SIZE
    pool = CachePool(device, slots=3, context=chunk_size * 3, sp=sp, bundles_per_bank=128, chunk_size=chunk_size)
    pool.update(0, 0, chunk_size)
    before = pool.check([chunk_pages, 0, 0])[0].clone()
    expected_chunk = torch.arange(local_chunk_pages, dtype=torch.int32).repeat(sp)
    assert torch.equal(before[0, :chunk_pages], expected_chunk)
    if sp == 8 and local_chunk_pages == 20:
        # The agreed 5120-token example: each SP gets 20 consecutive pages, B0..B19.
        for owner_sp in range(8):
            assert before[0, owner_sp * 20 : (owner_sp + 1) * 20].tolist() == list(range(20))

    # Grow across a partial page, a device boundary, then a chunk boundary.
    live_pages = chunk_pages
    for end in sorted({chunk_size + 1, chunk_size + local_chunk_pages * PAGE_SIZE + 1, 2 * chunk_size + 33}):
        pool.update(0, live_pages * PAGE_SIZE, end)
        live_pages = (end + PAGE_SIZE - 1) // PAGE_SIZE
        table = pool.check([live_pages, 0, 0])[0]
        assert torch.equal(table[0, :chunk_pages], before[0, :chunk_pages])
        # Fresh-pool allocation should be contiguous in local page order on each SP,
        # including chunk lengths that are not multiples of the number of banks.
        assert torch.equal(table[0, :live_pages], pool.local_page[:live_pages])
    snapshot = pool.check([live_pages, 0, 0])
    pool.update(0, chunk_size, end)
    assert all(torch.equal(a, b) for a, b in zip(snapshot, pool.check([live_pages, 0, 0])))

    pool.update(1, 0, chunk_size)
    before = pool.check([live_pages, chunk_pages, 0])[0].clone()
    pool.update(0, 0, 0)
    pool.update(2, 0, end)
    table = pool.check([0, chunk_pages, live_pages])[0]
    assert torch.equal(table[1], before[1])
    for owner_sp in range(sp):
        indices = torch.where(pool.owner[:live_pages] == owner_sp)[0]
        assert torch.equal(table[2, indices].sort().values, before[0, indices].sort().values)
    # Reset shorter, then reset larger using freed capacity while the other request stays live.
    for pages in (local_chunk_pages + 1, 2 * chunk_pages + 1):
        pool.update(2, 0, pages * PAGE_SIZE)
        table = pool.check([0, chunk_pages, pages])[0]
        assert torch.equal(table[1], before[1])


@pytest.mark.parametrize("chunk_size", [0, 1, 32, 5121, 2**32 - 1])
@pytest.mark.parametrize("tensor_requests", [False, True])
def test_invalid_chunk_size(device, expect_error, chunk_size, tensor_requests):
    pool = CachePool(device, sp=8)
    args = [request_tensor(value, device) for value in (0, 0, 32)] if tensor_requests else [0, 0, 32]
    with expect_error(RuntimeError, "chunk_size must be a positive multiple"):
        ttnn.experimental.update_cache_bundle_allocation(
            *pool.tensors, slot_id=args[0], actual_start=args[1], actual_end=args[2], chunk_size=chunk_size
        )
    pool.check([0, 0, 0])


def test_program_cache_distinguishes_chunk_sizes(device):
    device.enable_program_cache()
    # Identical tensor specs, different chunk geometry; neither program may be reused for the other.
    pools = [CachePool(device, context=10240, sp=8, bundles_per_bank=128, chunk_size=c) for c in (256, 5120)]
    entries = []
    for pool in pools:
        pool.update(0, 0, 5120)
        table = pool.check([160, 0, 0])[0]
        assert torch.equal(table[0, :160], pool.local_page[:160])
        entries.append(device.num_program_cache_entries())
    assert entries[1] == entries[0] + 1
    for pool in pools:
        pool.update(0, 5120, 10240)
        table = pool.check([320, 0, 0])[0]
        assert torch.equal(table[0, :320], pool.local_page[:320])
        assert device.num_program_cache_entries() == entries[1]
