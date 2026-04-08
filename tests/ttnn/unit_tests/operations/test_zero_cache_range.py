# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ttnn.kv_cache.zero_cache_range operation.
Verifies that a specified token range in a KV cache tensor is zeroed out
while leaving other regions untouched.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "seq_len, head_dim, start_token, end_token",
    [
        (3200, 576, 40, 128),  # partial first tile row, 3 tile rows zeroed (32-127)
        (3200, 576, 0, 32),  # zero first chunk
        (3200, 576, 64, 128),  # zero 2 tile rows (64-127)
        (3200, 576, 50, 256),  # larger range across multiple tile rows
        (3200, 576, 32, 64),  # exactly one tile row
        (640, 576, 33, 128),  # smaller seq_len, non-aligned start
    ],
)
def test_zero_cache_range(device, seq_len, head_dim, start_token, end_token):
    """
    Test that zero_cache_range zeroes the correct tile-aligned region
    and leaves other regions unchanged.
    """
    TILE_HEIGHT = 32

    # Create cache filled with ones so we can detect zeroing
    torch_cache = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)

    # Compute expected tile-aligned boundaries
    tile_start = (start_token // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = ((end_token + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len)

    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run zero_cache_range
    tt_cache = ttnn.kv_cache.zero_cache_range(tt_cache, start_token, end_token)

    # Read back
    result = ttnn.to_torch(tt_cache).to(torch.float32)

    # Check region before zeroed range is unchanged
    if tile_start > 0:
        before = result[:, :, :tile_start, :]
        assert torch.all(before != 0), f"Region before tile_start={tile_start} should be non-zero"

    # Check zeroed region
    zeroed = result[:, :, tile_start:tile_end, :]
    assert torch.all(zeroed == 0), (
        f"Region [{tile_start}:{tile_end}] should be all zeros, "
        f"but has {(zeroed != 0).sum().item()} non-zero elements"
    )

    # Check region after zeroed range is unchanged
    if tile_end < seq_len:
        after = result[:, :, tile_end:, :]
        assert torch.all(after != 0), f"Region after tile_end={tile_end} should be non-zero"


@pytest.mark.parametrize(
    "seq_len, head_dim, start_token, end_token",
    [
        (3200, 576, 40, 128),
    ],
)
def test_zero_cache_range_nd_sharded(device, seq_len, head_dim, start_token, end_token):
    """
    Test zero_cache_range with ND_SHARDED DRAM memory config (matching production KV cache layout).
    """
    TILE_HEIGHT = 32
    BH_NUM_DRAM_BANKS = 8
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32

    torch_cache = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)

    tile_start = (start_token // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = ((end_token + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len)

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)

    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )

    tt_cache = ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
    )

    tt_cache = ttnn.kv_cache.zero_cache_range(tt_cache, start_token, end_token)

    result = ttnn.to_torch(tt_cache).to(torch.float32)

    if tile_start > 0:
        before = result[:, :, :tile_start, :]
        assert torch.all(before != 0), f"Region before tile_start={tile_start} should be non-zero"

    zeroed = result[:, :, tile_start:tile_end, :]
    assert torch.all(zeroed == 0), (
        f"Region [{tile_start}:{tile_end}] should be all zeros, "
        f"but has {(zeroed != 0).sum().item()} non-zero elements"
    )

    if tile_end < seq_len:
        after = result[:, :, tile_end:, :]
        assert torch.all(after != 0), f"Region after tile_end={tile_end} should be non-zero"
