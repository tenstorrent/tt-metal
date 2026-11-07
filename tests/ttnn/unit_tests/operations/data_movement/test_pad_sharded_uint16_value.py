# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("cores", [1, 2])
@pytest.mark.parametrize("pad_value", [0, 3, 257, 65535])
def test_pad_sharded_uint16_width_value(device, cores, pad_value):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})

    def memory(width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32 // cores, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    host = (torch.arange(32 * 96, dtype=torch.int32) % 251).reshape(1, 1, 32, 96)
    source = ttnn.from_torch(
        host, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory(96)
    )
    for _ in range(2):
        output = ttnn.pad(
            source, [1, 1, 32, 112], [0, 0, 0, 0], pad_value, use_multicore=True, memory_config=memory(112)
        )
        expected = torch.full((1, 1, 32, 112), pad_value, dtype=torch.int32)
        expected[..., :96] = host
        assert torch.equal(ttnn.to_torch(output).to(torch.int32), expected)
        assert torch.equal(ttnn.to_torch(source).to(torch.int32), host)


@pytest.mark.parametrize("cores", [1, 2])
def test_pad_sharded_bfloat16_width_value(device, cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})

    def memory(width):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32 // cores, width], ttnn.ShardOrientation.ROW_MAJOR),
        )

    host = torch.ones((1, 1, 32, 96), dtype=torch.bfloat16)
    source = ttnn.from_torch(host, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory(96))
    output = ttnn.pad(source, [1, 1, 32, 112], [0, 0, 0, 0], 3, use_multicore=True, memory_config=memory(112))
    expected = torch.full((1, 1, 32, 112), 3, dtype=torch.bfloat16)
    expected[..., :96] = host
    assert torch.equal(ttnn.to_torch(output), expected)
