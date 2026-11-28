# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


NH, HD = 32, 128


def _sdpa_output_mem_config():
    return ttnn.create_sharded_memory_config(
        shape=(32, HD),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("pad_tiles", [0, 5])
def test_concat_heads_decode_matches_nlp_concat_heads_decode(device, pad_tiles):
    torch.manual_seed(0)
    pad = (
        ttnn.from_torch(
            torch.zeros(1, 1, 32, 32 * pad_tiles),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if pad_tiles
        else None
    )
    entries = None
    for x in (
        torch.randn(1, 1, NH, HD),
        (torch.arange(NH * HD, dtype=torch.float32) % 2039).reshape(1, 1, NH, HD) - 1000,
    ):
        x = x.bfloat16()
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sharded = ttnn.to_memory_config(t, _sdpa_output_mem_config())
        ref = ttnn.experimental.nlp_concat_heads_decode(sharded, num_heads=NH)
        out = ttnn.experimental.llama31_8b.concat_heads_decode(t, NH)
        if entries is None:
            entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == entries
        assert out.spec == ref.spec
        got = ttnn.to_torch(out)[0, 0, 0]
        assert torch.equal(got.view(torch.int16), ttnn.to_torch(ref)[0, 0, 0].view(torch.int16))
        assert torch.equal(got.bfloat16(), x[0, 0].reshape(-1))
    if pad is not None:
        ttnn.deallocate(pad)
