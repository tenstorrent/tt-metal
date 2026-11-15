# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.all_gather_for_matmul`` on DSpark blocks in (h x 32) tiles: the replica keeps the
input's tiles (no untilize), each width shard is appended as one contiguous run.

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_all_gather_for_matmul_rows.py
"""

import pytest
import torch

import ttnn

D = 5120
HEADS, HEAD_DIM = 16, 512
COLLAPSE = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
CORE00 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})


def _full_grid(device):
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})


def _mem(layout, grid, shard):
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


def _token_block(device, rows, h, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, 1, rows, D, dtype=torch.bfloat16)
    tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((h, 32)),
        device=device,
        memory_config=_mem(ttnn.TensorMemoryLayout.WIDTH_SHARDED, COLLAPSE, [h, D // 8]),
    )
    return x, tt


def _attn_block(device, rows, h, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, 1, rows, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((h, 32)),
        device=device,
        memory_config=_mem(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, CORE00, [h, HEADS * HEAD_DIM]),
    )
    return x, tt


def _check(out_tt, x, h, num_cores):
    rows, width = x.shape[-2], x.shape[-1]
    assert out_tt.layout == ttnn.TILE_LAYOUT and tuple(out_tt.tile.tile_shape) == (h, 32)
    assert out_tt.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert tuple(out_tt.memory_config().shard_spec.shape) == (h, width)
    assert tuple(out_tt.shape) == (1, 1, num_cores * h, width), out_tt.shape
    out = ttnn.to_torch(out_tt).reshape(num_cores, h, width)
    for core in range(num_cores):
        for j in range(rows):
            assert torch.equal(out[core, j], x[0, 0, j]), f"core {core} row {j}"


@pytest.mark.parametrize("rows, h", [(4, 4), (1, 1), (5, 8), (2, 2)], ids=str)
@pytest.mark.parametrize("kind", ["token", "attn"])
@pytest.mark.parametrize("out_grid", ["full", "4x4"])
def test_all_gather_for_matmul_rows(device, rows, h, kind, out_grid):
    x, x_tt = (_token_block if kind == "token" else _attn_block)(device, rows, h)
    grid = (
        _full_grid(device)
        if out_grid == "full"
        else ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(4, 4))})
    )
    for _ in range(2):
        out_tt = ttnn.experimental.deepseek.all_gather_for_matmul(x_tt, grid)
    _check(out_tt, x, h, grid.num_cores())


def test_all_gather_for_matmul_rows_perf(device):
    """s = 1: today's ROW_MAJOR row ([1, D] width-sharded over the collapse cores; o_a's [H, Dh] on core (0, 0)).
    s = 4: the (4 x 32) blocks. Both multicast onto the whole grid's bounding box."""
    from tracy import signpost

    grid = _full_grid(device)
    torch.manual_seed(0)
    tok1 = ttnn.from_torch(
        torch.randn(1, 1, 1, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_mem(ttnn.TensorMemoryLayout.WIDTH_SHARDED, COLLAPSE, [1, D // 8]),
    )
    attn1 = ttnn.from_torch(
        torch.randn(1, 1, HEADS, HEAD_DIM, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_mem(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, CORE00, [HEADS, HEAD_DIM]),
    )
    _, tok4 = _token_block(device, 4, 4)
    _, attn4 = _attn_block(device, 4, 4)
    cases = {"token_s1": tok1, "token_s4": tok4, "attn_s1": attn1, "attn_s4": attn4}
    for name, x in cases.items():
        ttnn.deallocate(ttnn.experimental.deepseek.all_gather_for_matmul(x, grid))
        ttnn.synchronize_device(device)
        signpost(name)
        for _ in range(20):
            ttnn.deallocate(ttnn.experimental.deepseek.all_gather_for_matmul(x, grid))
        ttnn.synchronize_device(device)
