# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 — retile_tile_input: TILE input at one tile height -> TILE output at another."""

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize

HEIGHTS = [32, 16, 8, 4, 2, 1]


def _retile(device, shape, in_h, out_h, mem_cfg=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_cfg,
        tile=ttnn.Tile([in_h, 32]),
    )
    y = tilize(t, memory_config=mem_cfg, tile=ttnn.Tile([out_h, 32]))
    assert y.layout == ttnn.TILE_LAYOUT
    assert tuple(y.tile.tile_shape) == (out_h, 32)
    assert torch.equal(ttnn.to_torch(y), x)


@pytest.mark.parametrize("in_h", HEIGHTS)
@pytest.mark.parametrize("out_h", HEIGHTS)
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 16, 128), (2, 3, 40, 96), (1, 1, 2048, 64)])
def test_retile_interleaved(device, shape, in_h, out_h):
    _retile(device, shape, in_h, out_h)


@pytest.mark.parametrize("in_h,out_h", [(32, 16), (1, 32), (16, 2)])
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 1, 32, 1024), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 1, 1024, 32), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((1, 1, 256, 256), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
def test_retile_sharded(device, shape, layout, in_h, out_h):
    H, W = shape[-2], shape[-1]
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(W // 32 - 1, H // 32 - 1))})
    else:
        n = H // 32 if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else W // 32
        grid = ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), row_wise=True)
    mem = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR))
    _retile(device, shape, in_h, out_h, mem)
