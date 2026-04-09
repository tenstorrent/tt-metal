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
from models.demos.deepseek_v3_d_p.tt.mla.utils import zero_cache_padding_zigzag


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
    "mesh_device",
    [
        (2, 4),
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
    """Test zero_cache_padding_zigzag on a 2x4 mesh with chunk boundary crossing.

    With sp_factor=2, seq_len=6400: chunk_size=1600.
    global_end_token=1580, padded_end=1664.
    Padding crosses the chunk 0/1 boundary at 1600, so device 0 and device 1
    both get zeroing.

    Device 0 should have local tokens [1568, 1600) zeroed.
    Device 1 should have local tokens [0, 64) zeroed.
    All other devices (TP replicas) should be entirely untouched.
    """
    sp_axis = 0
    sp_factor = mesh_device.shape[sp_axis]  # 2
    head_dim = 576
    seq_len = sp_factor * 3200  # 6400, gives chunk_size=1600
    seq_len_local = seq_len // sp_factor  # 3200
    global_end_token = 1580
    decode_chunk_align = 128

    # Create mesh cache filled with ones, replicated across all devices
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

    # Run zigzag padding — dispatches to SP devices 0 and 1, across all TP replicas
    tp_factor = mesh_device.shape[1]
    zero_cache_padding_zigzag(tt_cache, global_end_token, sp_factor, seq_len, decode_chunk_align, tp_factor)

    # Read back per-device tensors and verify
    # Mesh is 2x4 (SP x TP), so device_tensors has 8 entries.
    # SP device 0 = indices 0,1,2,3 (TP replicas), SP device 1 = indices 4,5,6,7
    device_tensors = ttnn.get_device_tensors(tt_cache)
    num_devices = len(device_tensors)
    tp_factor = mesh_device.shape[1]

    for dev_idx in range(num_devices):
        result = ttnn.to_torch(device_tensors[dev_idx]).to(torch.float32)
        sp_dev = dev_idx // tp_factor

        if sp_dev == 0:
            # SP device 0: local tokens [1568, 1600) should be zero
            before = result[:, :, :1568, :]
            assert torch.all(before != 0), f"Device {dev_idx} (SP 0): region before 1568 should be non-zero"

            zeroed = result[:, :, 1568:1600, :]
            assert torch.all(zeroed == 0), (
                f"Device {dev_idx} (SP 0): region [1568:1600] should be zeros, "
                f"but has {(zeroed != 0).sum().item()} non-zero elements"
            )

            after = result[:, :, 1600:, :]
            assert torch.all(after != 0), f"Device {dev_idx} (SP 0): region after 1600 should be non-zero"

        elif sp_dev == 1:
            # SP device 1: local tokens [0, 64) should be zero
            zeroed = result[:, :, :64, :]
            assert torch.all(zeroed == 0), (
                f"Device {dev_idx} (SP 1): region [0:64] should be zeros, "
                f"but has {(zeroed != 0).sum().item()} non-zero elements"
            )

            after = result[:, :, 64:, :]
            assert torch.all(after != 0), f"Device {dev_idx} (SP 1): region after 64 should be non-zero"
