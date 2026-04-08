# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ttnn.kv_cache.zero_cache_range operation.
Verifies that a specified token range in a KV cache tensor is zeroed out
while leaving other regions untouched.
Uses ND_SHARDED DRAM memory config matching production KV cache layout.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id, zero_cache_padding_zigzag


BH_NUM_DRAM_BANKS = 8
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def create_nd_sharded_cache(device, seq_len, head_dim):
    """Create an ND_SHARDED KV cache tensor filled with ones."""
    torch_cache = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)

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

    return ttnn.from_torch(
        torch_cache,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
    )


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

    # Compute expected tile-aligned boundaries
    tile_start = (start_token // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = ((end_token + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len)

    tt_cache = create_nd_sharded_cache(device, seq_len, head_dim)

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


def test_zigzag_mapping_logic():
    """Test that global_to_local_token_id correctly maps tokens under zigzag attention."""
    sp_factor = 4
    seq_len = 3200
    num_chunks = 2 * sp_factor  # 8
    chunk_size = seq_len // num_chunks  # 400

    # Device 0 holds chunks 0 (global 0-399) and 7 (global 2800-3199)
    dev, local = global_to_local_token_id(0, sp_factor, seq_len)
    assert dev == 0 and local == 0

    dev, local = global_to_local_token_id(399, sp_factor, seq_len)
    assert dev == 0 and local == 399

    # Chunk 7 -> device 0, local offset = chunk_size + offset
    dev, local = global_to_local_token_id(2800, sp_factor, seq_len)
    assert dev == 0 and local == chunk_size

    dev, local = global_to_local_token_id(3199, sp_factor, seq_len)
    assert dev == 0 and local == 2 * chunk_size - 1

    # Device 1 holds chunks 1 (global 400-799) and 6 (global 2400-2799)
    dev, local = global_to_local_token_id(400, sp_factor, seq_len)
    assert dev == 1 and local == 0

    dev, local = global_to_local_token_id(2400, sp_factor, seq_len)
    assert dev == 1 and local == chunk_size


@pytest.mark.parametrize(
    "global_end_token, decode_chunk_align",
    [
        (50, 128),
        (400, 128),
    ],
)
def test_zero_cache_padding_zigzag_single_device(device, global_end_token, decode_chunk_align):
    """Test zero_cache_padding_zigzag on a single device (sp_factor=1)."""
    TILE_HEIGHT = 32
    sp_factor = 1
    seq_len_local = 3200
    seq_len = seq_len_local * sp_factor
    head_dim = 576

    tt_cache = create_nd_sharded_cache(device, seq_len_local, head_dim)

    zero_cache_padding_zigzag(tt_cache, global_end_token, sp_factor, seq_len, decode_chunk_align)

    result = ttnn.to_torch(tt_cache).to(torch.float32)

    padded_end = ((global_end_token + decode_chunk_align - 1) // decode_chunk_align) * decode_chunk_align
    padded_end = min(padded_end, seq_len)

    # For sp_factor=1, zigzag is trivial: device 0 holds everything
    # Tile-aligned start of zeroed region
    tile_start = (global_end_token // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = ((padded_end + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    tile_end = min(tile_end, seq_len_local)

    if tile_start > 0:
        before = result[:, :, :tile_start, :]
        assert torch.all(before != 0), f"Region before tile_start={tile_start} should be non-zero"

    zeroed = result[:, :, tile_start:tile_end, :]
    assert torch.all(zeroed == 0), (
        f"Region [{tile_start}:{tile_end}] should be all zeros, "
        f"but has {(zeroed != 0).sum().item()} non-zero elements"
    )

    if tile_end < seq_len_local:
        after = result[:, :, tile_end:, :]
        assert torch.all(after != 0), f"Region after tile_end={tile_end} should be non-zero"
