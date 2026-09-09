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

    def __init__(self, device, *, slots=3, context=288, sp=2, bundles=3):
        self.sp, self.bundles = sp, bundles
        pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
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
                torch.arange(bundles - 1, -1, -1, dtype=torch.int32).repeat(sp, 1),
                torch.full((1, sp), bundles, dtype=torch.int32),
            )
        ]

    def update(self, slot, start, end):
        return ttnn.experimental.update_cache_bundle_allocation(
            *self.tensors, slot_id=slot, actual_start=start, actual_end=end, page_size=PAGE_SIZE
        )

    def check(self, pages_per_slot):
        state = []
        for name, tensor in zip(("page_table", "allocated_pages", "free_list", "free_count"), self.tensors):
            replicas = [ttnn.to_torch(shard).to(torch.int32) for shard in ttnn.get_device_tensors(tensor)]
            assert all(torch.equal(replica, replicas[0]) for replica in replicas), f"Replicas disagree on {name}"
            state.append(replicas[0])
        table, allocated, free, counts = state
        assert allocated[0].tolist() == pages_per_slot
        for sp in range(self.sp):
            live = torch.cat([table[slot, sp : pages : self.sp] for slot, pages in enumerate(pages_per_slot)])
            available = int(counts[0, sp])
            assert available == self.bundles - len(live), f"Wrong free count on SP{sp}"
            owners = torch.cat((live, free[sp, :available])).sort().values
            assert torch.equal(owners, torch.arange(self.bundles, dtype=torch.int32)), f"Invalid ownership on SP{sp}"
        return state


def test_concurrent_requests_and_slot_reuse(device):
    device.enable_program_cache()
    pools = [CachePool(device) for _ in range(2)]  # Distinct live buffers exercise program-cache reuse.
    for pool_index, pool in enumerate(pools):
        # slot, token start/end, expected live bundle IDs for each slot.
        requests = [
            (0, 0, 33, [[0, 0], [], []]),  # Prefill.
            (0, 33, 64, [[0, 0], [], []]),  # Decode within allocated pages.
            (0, 64, 65, [[0, 0, 1], [], []]),  # Decode across a page boundary.
            (0, 64, 65, [[0, 0, 1], [], []]),  # Repeated growth is idempotent.
            (1, 0, 33, [[0, 0, 1], [2, 1], []]),  # A concurrent request.
            (0, 0, 0, [[], [2, 1], []]),  # Completion returns bundles to the pool.
            (2, 0, 65, [[], [2, 1], [1, 0, 0]]),  # Another request reuses them.
            (2, 0, 33, [[], [2, 1], [0, 0]]),  # Replace a slot with a shorter prompt.
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


def test_full_pool_release_and_reuse(device, expect_error):
    pool = CachePool(device, bundles=1)
    result = pool.update(0, 0, 64)
    assert result.buffer_address() == pool.tensors[0].buffer_address()
    before = pool.check([2, 0, 0])
    pool.update(0, 0, 64)  # A full pool can reuse the replacing slot's own bundles.
    pool.check([2, 0, 0])
    with expect_error(RuntimeError, "actual_start"):
        pool.update(1, 64, 32)  # Invalid token ranges are rejected on the host.
    assert all(torch.equal(old, new) for old, new in zip(before, pool.check([2, 0, 0])))
    pool.update(0, 0, 0)
    pool.check([0, 0, 0])
    pool.update(1, 0, 64)
    table, _, _, _ = pool.check([0, 2, 0])
    assert table[1, :2].tolist() == [0, 0]


@pytest.mark.parametrize(
    "slots,context,sp",
    [(500, 1_048_576, 8), (500, 1_048_576, 32), (1, 65_537 * PAGE_SIZE, 1)],
    ids=["500_users_1m_context_sp8", "500_users_1m_context_sp32", "single_context_over_65535_pages"],
)
def test_large_contexts(device, slots, context, sp):
    pages = context // PAGE_SIZE
    bundles = slots * pages // sp
    pool = CachePool(device, slots=slots, context=context, sp=sp, bundles=bundles)
    for slot in range(slots):
        pool.update(slot, 0, context)
    table, _, _, _ = pool.check([pages] * slots)
    expected = torch.arange(bundles, dtype=torch.int32).reshape(slots, pages // sp).repeat_interleave(sp, dim=1)
    assert torch.equal(table, expected)

    # Replace a full context using its own bundles, preserving every other slot.
    slot = slots // 2
    pool.update(slot, 0, context)
    table, _, _, _ = pool.check([pages] * slots)
    expected[slot] = expected[slot].reshape(-1, sp).flip(0).flatten()
    assert torch.equal(table, expected)

    for slot in range(slots):
        pool.update(slot, 0, 0)
    pool.check([0] * slots)  # Every bundle is available again, including IDs above UINT16.


@pytest.mark.parametrize("sp", [1, 8, 17, 32, 128])
def test_chunked_growth_preserves_other_slots(device, sp):
    # Large table rows, unaligned slot counters and free-list tails, and 4 KiB crossings.
    pool = CachePool(device, slots=19, context=1_048_576, sp=sp, bundles=2051)
    pages = [0] * 19
    for slot in (0, 16, 18):
        pool.update(slot, 0, 33)
        pages[slot] = 2
    before = pool.check(pages)[0].clone()
    for end_pages in (160, 161, 320, 1023, 1025, 1600):
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
    pools = [CachePool(device, bundles=8) for _ in range(2)]
    request_buffers = []
    entries = None
    for pool in pools:
        reference = CachePool(device, bundles=8)
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
    pool = CachePool(device, context=10240, sp=8, bundles=128)
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
