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


@pytest.mark.parametrize(
    "global_end_token, decode_chunk_align",
    [
        (50, 128),
        (400, 128),
        (1580, 128),  # edge case: padding crosses zigzag chunk boundary (chunk_size=1600 for sp=1, seq=3200)
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


def test_zigzag_chunk_boundary_crossing():
    """Test that padding near a zigzag chunk boundary correctly maps to two devices.

    With sp_factor=32, seq_len=102400: chunk_size=1600.
    global_end_token=1580, padded_end=1664.
    Padding range [1568, 1664) (tile-aligned) crosses the chunk 0/1 boundary at 1600,
    so device 0 and device 1 both need zeroing.
    """
    import math

    sp_factor = 32
    seq_len = 102400
    global_end_token = 1580
    decode_chunk_align = 128
    tile_size = 32

    padded_end = math.ceil(global_end_token / decode_chunk_align) * decode_chunk_align
    global_start = (global_end_token // tile_size) * tile_size

    # Manually compute device_ranges (same logic as zero_cache_padding_zigzag)
    device_ranges: dict[int, tuple[int, int]] = {}
    for global_tok in range(global_start, padded_end, tile_size):
        device_id, local_token_id = global_to_local_token_id(global_tok, sp_factor, seq_len)
        local_tile_start = (local_token_id // tile_size) * tile_size
        local_tile_end = local_tile_start + tile_size

        if device_id not in device_ranges:
            device_ranges[device_id] = (local_tile_start, local_tile_end)
        else:
            prev_start, prev_end = device_ranges[device_id]
            device_ranges[device_id] = (min(prev_start, local_tile_start), max(prev_end, local_tile_end))

    # Should span two devices
    assert len(device_ranges) == 2, f"Expected 2 devices, got {len(device_ranges)}: {device_ranges}"
    assert 0 in device_ranges, "Device 0 should be in the range"
    assert 1 in device_ranges, "Device 1 should be in the range"

    # Device 0: chunk 0 tail (global 1568-1599 -> local 1568-1599)
    assert device_ranges[0][0] == 1568
    assert device_ranges[0][1] == 1600

    # Device 1: chunk 1 start (global 1600-1663 -> local 0-63)
    assert device_ranges[1][0] == 0
    assert device_ranges[1][1] == 64


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_zero_cache_padding_zigzag_multi_device(mesh_device):
    """Test zero_cache_padding_zigzag on a multi-device mesh with chunk boundary crossing.

    With sp_factor=32, seq_len=102400: chunk_size=1600.
    global_end_token=1580, padded_end=1664.
    Padding crosses the chunk 0/1 boundary at 1600, so device 0 and device 1
    both get zeroing.

    Device 0 should have local tokens [1568, 1600) zeroed.
    Device 1 should have local tokens [0, 64) zeroed.
    All other devices should be entirely untouched.
    """
    TILE_HEIGHT = 32
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    head_dim = 576
    seq_len = 102400
    seq_len_local = seq_len // sp_factor
    global_end_token = 1580
    decode_chunk_align = 128

    # Create mesh cache filled with ones
    torch_cache = torch.ones(1, 1, seq_len_local, head_dim, dtype=torch.bfloat16)

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
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run zigzag padding
    zero_cache_padding_zigzag(tt_cache, global_end_token, sp_factor, seq_len, decode_chunk_align)

    # Read back per-device tensors and verify
    device_tensors = ttnn.get_device_tensors(tt_cache)

    for dev_idx in range(len(device_tensors)):
        result = ttnn.to_torch(device_tensors[dev_idx]).to(torch.float32)

        if dev_idx == 0:
            # Device 0: local tokens [1568, 1600) should be zero
            before = result[:, :, :1568, :]
            assert torch.all(before != 0), f"Device 0: region before 1568 should be non-zero"

            zeroed = result[:, :, 1568:1600, :]
            assert torch.all(zeroed == 0), (
                f"Device 0: region [1568:1600] should be zeros, "
                f"but has {(zeroed != 0).sum().item()} non-zero elements"
            )

            after = result[:, :, 1600:, :]
            assert torch.all(after != 0), f"Device 0: region after 1600 should be non-zero"

        elif dev_idx == 1:
            # Device 1: local tokens [0, 64) should be zero
            zeroed = result[:, :, :64, :]
            assert torch.all(zeroed == 0), (
                f"Device 1: region [0:64] should be zeros, " f"but has {(zeroed != 0).sum().item()} non-zero elements"
            )

            after = result[:, :, 64:, :]
            assert torch.all(after != 0), f"Device 1: region after 64 should be non-zero"

        else:
            # All other devices should be entirely non-zero (untouched)
            assert torch.all(result != 0), f"Device {dev_idx}: should be entirely non-zero (untouched)"
