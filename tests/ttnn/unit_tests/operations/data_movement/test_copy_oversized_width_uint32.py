# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _make_oversized_width_input(device, offset):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [4, 5], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Include high bits and low bits so a floating-point conversion cannot pass.
    words = torch.arange(12, dtype=torch.int64).reshape(1, 4, 3) * 100000003 + 0x80000001 + offset
    host = words.to(torch.int32)
    tensor = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory)
    assert torch.equal(ttnn.to_torch(tensor).to(torch.int32), host)
    return tensor, host


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("output_memory", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_to_memory_config_oversized_width_uint32(device, output_memory):
    for offset in (0, 37):
        source, host = _make_oversized_width_input(device, offset)
        result = ttnn.to_memory_config(source, output_memory)
        assert result.memory_config() == output_memory
        assert torch.equal(ttnn.to_torch(result).to(torch.int32), host)
        assert torch.equal(ttnn.to_torch(source).to(torch.int32), host)


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize("pattern", ["bf16-block-to-width", "uint32-width-to-block"])
def test_to_memory_config_both_shard_rows_oversized(device, pattern):
    single_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    for offset in (0, 37):
        if pattern == "bf16-block-to-width":
            host = (torch.arange(7).reshape(1, 7) * 0.25 - 1.75 + offset).to(torch.bfloat16)
            input_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(single_core, [1, 63], ttnn.ShardOrientation.ROW_MAJOR),
            )
            source = ttnn.from_torch(
                host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
            )
            output_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(single_core, [1, 21], ttnn.ShardOrientation.ROW_MAJOR),
            )
            word_dtype = torch.int16
        else:
            source, host = _make_oversized_width_input(device, offset)
            output_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
            output_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(output_grid, [1, 63], ttnn.ShardOrientation.ROW_MAJOR),
            )
            word_dtype = torch.int32

        expected_words = host.view(word_dtype)
        assert torch.equal(ttnn.to_torch(source).view(word_dtype), expected_words)
        result = ttnn.to_memory_config(source, output_memory)
        actual_words = ttnn.to_torch(result).view(word_dtype)
        input_words = ttnn.to_torch(source).view(word_dtype)
        assert result.memory_config() == output_memory
        assert torch.equal(
            input_words, expected_words
        ), f"{torch.count_nonzero(input_words != expected_words)} input words changed"
        assert torch.equal(
            actual_words, expected_words
        ), f"{torch.count_nonzero(actual_words != expected_words)} output words differ"
