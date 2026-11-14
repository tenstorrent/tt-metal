# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def dram_width_sharded(device, k, shard_tiles):
    grid = device.dram_grid_size()
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
        (k, shard_tiles * 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


# The 2D multicast factory must read B with the shard's physical width. A shard wider than ceil(N / banks)
# tiles (e.g. decode weights padded so each bank splits evenly across readers) leaves the last bank short.
# Gemma 4 31B down projection: N = 168 tiles over 8 banks, shard 22 tiles (2 readers x 11), last bank 14 tiles.
@pytest.mark.parametrize("shard_extra_tiles", [0, 1], ids=["exact", "padded"])
@pytest.mark.parametrize("m, k, n", [(128, 1024, 5376), (128, 1024, 3072)])
def test_matmul_2d_dram_sharded_in1_shard_width(device, m, k, n, shard_extra_tiles):
    torch.manual_seed(0)
    banks = device.dram_grid_size().x * device.dram_grid_size().y
    n_tiles = n // 32
    shard_tiles = math.ceil(n_tiles / banks) + shard_extra_tiles
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, min(grid.y, m // 32)

    torch_x = torch.randn(1, 1, m, k)
    torch_w = torch.randn(1, 1, k, n) * 0.02
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w_sharded = ttnn.from_torch(
        torch_w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_width_sharded(device, k, shard_tiles),
    )
    w_interleaved = ttnn.from_torch(
        torch_w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    per_core_n = math.ceil(n_tiles / gx)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=max(i for i in range(1, min(per_core_n, 8) + 1) if per_core_n % i == 0),
        per_core_M=math.ceil(m // 32 / gy),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )

    def run(w):
        out = ttnn.linear(x, w, program_config=program_config, dtype=ttnn.bfloat16)
        return ttnn.to_torch(out)

    # A wrong B stride reads the wrong columns, which drops PCC far below these bounds. Interleaved B takes a
    # different reader, so the two results agree closely but not bit for bit.
    expected = torch_x @ torch_w
    sharded = run(w_sharded)
    interleaved = run(w_interleaved)
    assert_with_pcc(expected, interleaved, 0.999)
    assert_with_pcc(expected, sharded, 0.999)
    assert_with_pcc(interleaved, sharded, 0.999)
