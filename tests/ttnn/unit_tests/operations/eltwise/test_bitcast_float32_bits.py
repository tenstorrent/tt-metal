# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [
        (ttnn.uint32, ttnn.float32),
        (ttnn.int32, ttnn.float32),
        (ttnn.float32, ttnn.uint32),
        (ttnn.float32, ttnn.int32),
        (ttnn.int32, ttnn.uint32),
        (ttnn.uint32, ttnn.int32),
    ],
)
@pytest.mark.parametrize(
    "layout,memory",
    [
        (ttnn.TILE_LAYOUT, "dram"),
        (ttnn.TILE_LAYOUT, "l1"),
        (ttnn.TILE_LAYOUT, "height_sharded"),
        (ttnn.TILE_LAYOUT, "block_sharded"),
        (ttnn.ROW_MAJOR_LAYOUT, "dram"),
        (ttnn.ROW_MAJOR_LAYOUT, "l1"),
        (ttnn.ROW_MAJOR_LAYOUT, "height_sharded"),
        (ttnn.ROW_MAJOR_LAYOUT, "block_sharded"),
    ],
)
def test_bitcast_preserves_all_32bit_words(device, source_dtype, target_dtype, layout, memory):
    generator = torch.Generator().manual_seed(61975)
    words = torch.randint(0, 1 << 32, (128 * 128,), generator=generator, dtype=torch.int64).to(torch.int32)
    patterns = torch.tensor(
        [
            0x00000000,
            0x00000001,
            0x007FFFFF,
            0x00800001,
            0x3F800001,
            0x32B779B1,
            0xBF800123,
            0x7F7FFFFF,
            0x80000000,
            0x80000001,
            0xFFFFFFFF,
            0x7FFFFFFF,
            0x7FC00001,
            0x7F800001,
            0xFF800001,
            0x7F800000,
            0xFF800000,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    words[: len(patterns)] = patterns
    words = words.reshape(128, 128)
    if memory in ("height_sharded", "block_sharded"):
        strategy = ttnn.ShardStrategy.HEIGHT if memory == "height_sharded" else ttnn.ShardStrategy.BLOCK
        memory_config = ttnn.create_sharded_memory_config(
            words.shape, core_grid=ttnn.CoreGrid(x=2, y=2), strategy=strategy
        )
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG if memory == "dram" else ttnn.L1_MEMORY_CONFIG
    host = words.view(torch.float32) if source_dtype == ttnn.float32 else words
    source = ttnn.from_torch(host, dtype=source_dtype, layout=layout, device=device, memory_config=memory_config)
    assert torch.equal(ttnn.to_torch(source).view(torch.int32), words), "Input upload changed stored words"
    output = ttnn.bitcast(source, target_dtype, memory_config=memory_config)
    assert output.dtype == target_dtype
    actual = ttnn.to_torch(output).view(torch.int32)
    assert torch.equal(actual, words), f"Bitcast changed {int((actual != words).sum())} stored 32-bit words"
