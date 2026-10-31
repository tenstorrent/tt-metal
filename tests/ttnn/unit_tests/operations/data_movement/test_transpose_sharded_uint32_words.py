# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32, ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("cores", [1, 2])
def test_transpose_sharded_preserves_words(device, layout, dtype, cores):
    shape = (2, 64, 32)
    words = torch.arange(2 * 64 * 32, dtype=torch.int64).reshape(shape)
    if dtype in (ttnn.uint32, ttnn.int32):
        host = ((words * 1000003) % (2**32)).to(torch.int32)
    else:
        host = (words % 127).to(torch.float32)
    memory = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(x=1, y=cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    source = ttnn.from_torch(host, dtype=dtype, layout=layout, device=device, memory_config=memory)

    def read(tensor):
        result = ttnn.to_torch(tensor)
        return result.to(torch.int64) & 0xFFFFFFFF if dtype in (ttnn.uint32, ttnn.int32) else result.float()

    expected_input = host.to(torch.int64) & 0xFFFFFFFF if dtype in (ttnn.uint32, ttnn.int32) else host
    assert torch.equal(read(source), expected_input)
    for _ in range(2):
        output = ttnn.transpose(source, -2, -1)
        assert torch.equal(read(output), expected_input.transpose(-2, -1))
        assert torch.equal(read(source), expected_input)


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32, ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("height", [16, 48, 144, 272])
def test_transpose_sharded_partial_height_preserves_words(device, dtype, height):
    # Row-major height shard [2 * height, 64] on one core; height is not a tile multiple.
    shape = (1, 2, height, 64)
    words = torch.arange(2 * height * 64, dtype=torch.int64).reshape(shape)
    if dtype == ttnn.float32:
        host = (words.to(torch.float32) - 12345) / 8192
    elif dtype == ttnn.bfloat16:
        host = (words % 251).to(torch.float32)  # Exact in bfloat16.
    else:
        host = ((words * 1000003) % (2**32)).to(torch.int32)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [2 * height, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    source = ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory)

    def read(tensor):
        result = ttnn.to_torch(tensor)
        return result.float() if dtype in (ttnn.float32, ttnn.bfloat16) else result.to(torch.int64) & 0xFFFFFFFF

    expected_input = host if dtype in (ttnn.float32, ttnn.bfloat16) else host.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(read(source), expected_input)
    for _ in range(2):
        output = ttnn.transpose(source, -2, -1)
        assert torch.equal(read(output), expected_input.transpose(-2, -1))
        assert torch.equal(read(source), expected_input)
