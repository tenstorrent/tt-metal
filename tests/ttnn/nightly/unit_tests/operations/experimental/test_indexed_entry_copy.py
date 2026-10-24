# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.indexed_entry_copy: one dim-0 entry copied in place, the entry chosen on device."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

# Validated on Blackhole only; the op itself is architecture-neutral.
pytestmark = run_for_blackhole()

# (dtype, layout, entry shape past dim 0): the FP32 tile and BF16 row-major forms the op serves.
FORMS = [
    pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, (3, 64, 64), id="fp32-tile"),
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (3, 192), id="bf16-row-major"),
]
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _device(tensor, device, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout, memory_config=memory_config)


def _index(device, value):
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _random(shape, dtype, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(TORCH_DTYPE[dtype])


def _nd_dram(device, shard_shape):
    banks = device.dram_grid_size().x
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=shard_shape,
            grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(banks)}
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


@pytest.mark.parametrize("dtype, layout, entry_shape", FORMS)
@pytest.mark.parametrize("scale, offset", [(1, 0), (3, 2)], ids=["plain", "scaled"])
def test_load_and_store_each_entry(device, dtype, layout, entry_shape, scale, offset):
    entries = 7
    table_host = _random((entries, *entry_shape), dtype, 1)
    table = _device(table_host, device, dtype, layout)
    single = _device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    for value in sorted({0, (entries - 1 - offset) // scale // 2, (entries - 1 - offset) // scale}):
        entry = value * scale + offset
        index = _index(device, value)

        ttnn.experimental.indexed_entry_copy(table, single, index, index_scale=scale, index_offset=offset)
        assert torch.equal(ttnn.to_torch(single), table_host[entry : entry + 1]), f"load entry {entry}"

        update = _random((1, *entry_shape), dtype, 100 + entry)
        source = _device(update, device, dtype, layout)
        ttnn.experimental.indexed_entry_copy(source, table, index, index_scale=scale, index_offset=offset)
        table_host[entry : entry + 1] = update
        assert torch.equal(ttnn.to_torch(table), table_host), f"store entry {entry} or a neighbour"
        ttnn.deallocate(source)
        ttnn.deallocate(index)


@pytest.mark.parametrize("dtype, layout, entry_shape", FORMS)
def test_entry_past_the_end_writes_nothing(device, dtype, layout, entry_shape):
    entries = 4
    table_host = _random((entries, *entry_shape), dtype, 2)
    table = _device(table_host, device, dtype, layout)
    single_host = _random((1, *entry_shape), dtype, 3)
    single = _device(single_host, device, dtype, layout)
    # The first is past the end before scaling, the second only after it.
    for value, scale, offset in [(entries, 1, 0), (1, 2, 2), (0xFFFFFFFF, 1, 0)]:
        index = _index(device, value)
        ttnn.experimental.indexed_entry_copy(table, single, index, index_scale=scale, index_offset=offset)
        ttnn.experimental.indexed_entry_copy(single, table, index, index_scale=scale, index_offset=offset)
        ttnn.deallocate(index)
    assert torch.equal(ttnn.to_torch(table), table_host)
    assert torch.equal(ttnn.to_torch(single), single_host)


@pytest.mark.parametrize("dtype, layout, entry_shape", FORMS)
def test_single_entry_on_both_sides_copies_at_entry_zero(device, dtype, layout, entry_shape):
    source_host = _random((1, *entry_shape), dtype, 4)
    source = _device(source_host, device, dtype, layout)
    destination_host = torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype])
    destination = _device(destination_host, device, dtype, layout)

    ttnn.experimental.indexed_entry_copy(source, destination, _index(device, 1))
    assert torch.equal(ttnn.to_torch(destination), destination_host)
    ttnn.experimental.indexed_entry_copy(source, destination, _index(device, 0))
    assert torch.equal(ttnn.to_torch(destination), source_host)


def test_store_into_nd_sharded_matches_fill_cache_for_user(device):
    """Native interleaved tiles into an ND-sharded [1, 1, D, 32] slab land where fill_cache_for_user_ puts them."""
    entries, heads, head_dim = 6, 3, 128
    memory_config = _nd_dram(device, [1, 1, head_dim, 32])
    slab_host = _random((entries, heads, head_dim, head_dim), ttnn.float32, 5)
    reference = _device(slab_host, device, ttnn.float32, ttnn.TILE_LAYOUT, memory_config)
    ours = _device(slab_host, device, ttnn.float32, ttnn.TILE_LAYOUT, memory_config)
    state_host = _random((1, heads, head_dim, head_dim), ttnn.float32, 6)
    state = _device(state_host, device, ttnn.float32, ttnn.TILE_LAYOUT)
    loaded = _device(torch.zeros_like(state_host), device, ttnn.float32, ttnn.TILE_LAYOUT)

    for batch in (0, 4, entries - 1):
        ttnn.kv_cache.fill_cache_for_user_(reference, state, batch)
        index = _index(device, batch)
        ttnn.experimental.indexed_entry_copy(state, ours, index)
        assert torch.equal(ttnn.to_torch(ours), ttnn.to_torch(reference)), f"batch {batch}"
        ttnn.experimental.indexed_entry_copy(ours, loaded, index)
        assert torch.equal(ttnn.to_torch(loaded), state_host), f"load back batch {batch}"
        ttnn.deallocate(index)


def test_row_store_matches_slice_write(device):
    """Whole row-major entries land where slice_write of the same rows puts them."""
    entries, rows, width = 5, 6, 192
    table_host = _random((entries, rows, width), ttnn.bfloat16, 7)
    reference = _device(table_host, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ours = _device(table_host, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    rows_host = _random((1, rows, width), ttnn.bfloat16, 8)
    update = _device(rows_host, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    for batch in (0, 2, entries - 1):
        ttnn.experimental.slice_write(update, reference, [batch, 0, 0], [batch + 1, rows, width], [1, 1, 1])
        index = _index(device, batch)
        ttnn.experimental.indexed_entry_copy(update, ours, index)
        assert torch.equal(ttnn.to_torch(ours), ttnn.to_torch(reference)), f"batch {batch}"
        ttnn.deallocate(index)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@pytest.mark.parametrize("dtype, layout, entry_shape", FORMS)
def test_one_trace_follows_the_index(device, dtype, layout, entry_shape):
    """A captured load + store pair addresses the entry written into the index before each replay."""
    entries, scale, offset = 6, 2, 1
    table_host = _random((entries, *entry_shape), dtype, 9)
    table = _device(table_host, device, dtype, layout)
    working = _device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    update = _device(torch.zeros(1, *entry_shape, dtype=TORCH_DTYPE[dtype]), device, dtype, layout)
    index = _index(device, 0)

    def step():
        ttnn.experimental.indexed_entry_copy(table, working, index, index_scale=scale, index_offset=offset)
        ttnn.experimental.indexed_entry_copy(update, table, index, index_scale=scale, index_offset=offset)

    step()  # compile before capture: index 0 stores the zero update into entry `offset`
    table_host[offset : offset + 1] = 0
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    step()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for value in (2, 0, 1, 2):
            entry = value * scale + offset
            expected_load = table_host[entry : entry + 1].clone()
            new = _random((1, *entry_shape), dtype, 200 + value)
            for destination, host in ((index, torch.tensor([value], dtype=torch.int64)), (update, new)):
                staged = ttnn.from_torch(
                    host,
                    device=device,
                    dtype=destination.dtype,
                    layout=destination.layout,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.copy(staged, destination)
                ttnn.deallocate(staged)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            table_host[entry : entry + 1] = new
            assert torch.equal(ttnn.to_torch(working), expected_load), f"replay load index {value}"
            assert torch.equal(ttnn.to_torch(table), table_host), f"replay store index {value}"
    finally:
        ttnn.release_trace(device, trace)


def test_rejects_two_multi_entry_operands(device, expect_error):
    source = _device(torch.zeros(2, 32, 32), device, ttnn.float32, ttnn.TILE_LAYOUT)
    destination = _device(torch.zeros(2, 32, 32), device, ttnn.float32, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "must hold a single entry"):
        ttnn.experimental.indexed_entry_copy(source, destination, _index(device, 0))


def test_rejects_mismatched_entry_shape(device, expect_error):
    source = _device(torch.zeros(4, 32, 64), device, ttnn.float32, ttnn.TILE_LAYOUT)
    destination = _device(torch.zeros(1, 32, 32), device, ttnn.float32, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "differ past dim 0"):
        ttnn.experimental.indexed_entry_copy(source, destination, _index(device, 0))
