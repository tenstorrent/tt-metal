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
@pytest.mark.parametrize("repeats", [(1, 1, 2), (1, 2, 2)], ids=["width", "height-and-width"])
def test_repeat_unshards_oversized_width_uint32(device, repeats):
    # A WIDTH-sharded last-dimension repeat cannot use the native sharded path.
    # Both calls must unshard through the row-major copy before repeating.
    for offset in (0, 37):
        source, host = _make_oversized_width_input(device, offset)
        result = ttnn.repeat(source, repeats, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert result.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert torch.equal(ttnn.to_torch(result).to(torch.int32), host.repeat(*repeats))
        assert torch.equal(ttnn.to_torch(source).to(torch.int32), host)
