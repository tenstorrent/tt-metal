# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


NQ, NKV, HD = 32, 8, 128
W = (NQ + 2 * NKV) * HD


def _out_mc():
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, HD),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _in_mc(layout):
    if layout == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    if layout == "l1_width_sharded_32_rows":
        grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
            ]
        )
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, W // 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, W // 32),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _as_int16(t):
    return ttnn.to_torch(t).contiguous().view(torch.int16)


@pytest.mark.parametrize("layout", ["l1_interleaved", "l1_width_sharded", "l1_width_sharded_32_rows"])
@pytest.mark.parametrize("pad_tiles", [0, 7])
def test_qkv_heads_decode_matches_nlp_create_qkv_heads_decode(device, layout, pad_tiles):
    torch.manual_seed(0)
    pad = (
        ttnn.from_torch(
            torch.zeros(1, 1, 32, 32 * pad_tiles),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if pad_tiles
        else None
    )
    entries = None
    for x in (torch.randn(1, 1, 1, W), (torch.arange(W, dtype=torch.float32) % 4093).reshape(1, 1, 1, W) - 2046.0):
        x = x.bfloat16()
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_in_mc(layout)
        )
        ref = ttnn.experimental.nlp_create_qkv_heads_decode(t, num_heads=NQ, num_kv_heads=NKV, memory_config=_out_mc())
        out = ttnn.experimental.llama31_8b.qkv_heads_decode(t, NQ, NKV, memory_config=_out_mc())
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        golden = (
            x[:, :, :, : HD * NQ].reshape(1, 1, NQ, HD),
            x[:, :, :, HD * NQ : HD * (NQ + NKV)].reshape(1, 1, NKV, HD),
            x[:, :, :, HD * (NQ + NKV) :].reshape(1, 1, NKV, HD),
        )
        for a, b, g in zip(ref, out, golden):
            assert a.spec == b.spec
            assert torch.equal(_as_int16(a), _as_int16(b))
            assert torch.equal(ttnn.to_torch(b).bfloat16()[:, :1], g)
    if pad is not None:
        ttnn.deallocate(pad)
