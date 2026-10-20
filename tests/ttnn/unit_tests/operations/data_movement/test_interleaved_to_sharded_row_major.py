# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _bits(t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t.to(torch.int64)


def _make(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    if dtype == ttnn.bfloat16:
        return torch.randn(shape, generator=g).to(torch.bfloat16)
    if dtype == ttnn.float32:
        return torch.randn(shape, generator=g, dtype=torch.float32)
    return torch.randint(0, 2**31 - 1, shape, generator=g, dtype=torch.int32)


ELEMENT_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint32: 4}


@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("grid", ["one_row", "full"])
@pytest.mark.parametrize("rows_per_core", [7, 100, 513])
@pytest.mark.parametrize("row_bytes", [32, 48, 128, 240, 256, 512])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint32])
def test_interleaved_to_sharded_row_major_dram(device, dtype, row_bytes, rows_per_core, grid, orientation):
    width = row_bytes // ELEMENT_BYTES[dtype]
    g = device.compute_with_storage_grid_size()
    grid_y = 1 if grid == "one_row" else g.y
    num_cores = g.x * grid_y
    rows = num_cores * rows_per_core - rows_per_core // 2
    shape = (1, 1, rows, width)

    torch_input = _make(shape, dtype, seed=row_bytes * 1000 + rows_per_core)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, grid_y - 1))})
    sharded_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (rows_per_core, width), orientation),
    )

    for out in (
        ttnn.to_memory_config(tt_input, sharded_config),
        ttnn.interleaved_to_sharded(tt_input, sharded_config),
    ):
        assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        assert torch.equal(_bits(ttnn.to_torch(out)), _bits(torch_input))
        ttnn.deallocate(out)
