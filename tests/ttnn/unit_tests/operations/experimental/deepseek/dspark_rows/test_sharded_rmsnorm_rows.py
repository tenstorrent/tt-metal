# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.sharded_rmsnorm`` on a DSpark block: rows x D in (h x 32) tiles, width-sharded
over the 8 collapse cores, normed per row.

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_sharded_rmsnorm_rows.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

D = 5120
EPS = 1e-6
GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})


def _sharded(shard_h, shard_w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(GRID, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _ref(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _block_inputs(device, rows, h, replicated_weight, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, 1, rows, D, dtype=torch.bfloat16) * (1 + torch.arange(rows).reshape(1, 1, rows, 1))
    w = torch.randn(1, 1, 1, D, dtype=torch.bfloat16)
    tile = ttnn.Tile((h, 32))
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=_sharded(h, D // 8)
    )
    if replicated_weight:
        w_tt = ttnn.from_torch(
            w.expand(1, 1, h, D).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        w_tt = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return x, w, x_tt, w_tt


@pytest.mark.parametrize("rows, h", [(4, 4), (1, 1), (2, 2), (5, 8), (8, 8)], ids=lambda v: str(v))
@pytest.mark.parametrize("replicated_weight", [True, False], ids=["w_rows", "w_tile32"])
@pytest.mark.parametrize("out_mode", ["sharded", "dram"])
def test_sharded_rmsnorm_rows(device, rows, h, replicated_weight, out_mode):
    x, w, x_tt, w_tt = _block_inputs(device, rows, h, replicated_weight)
    ref = _ref(x, w)
    mem = None if out_mode == "sharded" else ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):  # the second call runs from the program cache
        out_tt = ttnn.experimental.deepseek.sharded_rmsnorm(x_tt, w_tt, epsilon=EPS, memory_config=mem)
    assert out_tt.layout == ttnn.TILE_LAYOUT
    assert tuple(out_tt.tile.tile_shape) == (h, 32)
    if out_mode == "sharded":
        assert out_tt.memory_config() == x_tt.memory_config()
    out = ttnn.to_torch(out_tt).float()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    for j in range(rows):
        passing, pcc = comp_pcc(ref[..., j, :], out[..., j, :], 0.999)
        logger.info(f"row {j}: PCC {pcc}, max abs err {(ref[..., j, :] - out[..., j, :]).abs().max().item():.4f}")
        assert passing, f"row {j}: PCC {pcc}"


@pytest.mark.parametrize("rows, h", [(4, 4), (2, 2), (5, 8), (1, 1)], ids=lambda v: str(v))
@pytest.mark.parametrize("buffer", ["dram", "l1"])
def test_sharded_rmsnorm_rows_row_major(device, rows, h, buffer):
    """The block's rows un-tiled: ROW_MAJOR [1, 1, rows, D] interleaved, one page per row."""
    x, w, x_tt, w_tt = _block_inputs(device, rows, h, True)
    ref = _ref(x, w)
    mem = ttnn.DRAM_MEMORY_CONFIG if buffer == "dram" else ttnn.L1_MEMORY_CONFIG
    tiled = ttnn.to_torch(ttnn.experimental.deepseek.sharded_rmsnorm(x_tt, w_tt, epsilon=EPS)).float()
    for _ in range(2):  # the second call runs from the program cache
        out_tt = ttnn.experimental.deepseek.sharded_rmsnorm(
            x_tt, w_tt, epsilon=EPS, memory_config=mem, output_layout=ttnn.ROW_MAJOR_LAYOUT
        )
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(out_tt.shape) == (1, 1, rows, D)
    out = ttnn.to_torch(out_tt).float()
    # Bit-identical to the tiled output, row for row.
    assert torch.equal(out, tiled), (out - tiled).abs().max()
    for j in range(rows):
        passing, pcc = comp_pcc(ref[..., j, :], out[..., j, :], 0.999)
        assert passing, f"row {j}: PCC {pcc}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10240}], indirect=True)
def test_sharded_rmsnorm_rows_perf(device):
    """Production shape: D = 5120 on the 8 collapse cores. s = 1 is today's ROW_MAJOR row (ROW_MAJOR L1 output,
    the input norm); s = 4 the (4 x 32) block with the block's replicated gamma, TILE L1 output."""
    from tracy import signpost

    torch.manual_seed(0)
    x1 = torch.randn(1, 1, 1, D, dtype=torch.bfloat16)
    w = torch.randn(1, 1, 1, D, dtype=torch.bfloat16)
    x1_tt = ttnn.from_torch(
        x1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_sharded(1, D // 8)
    )
    w_tt = ttnn.from_torch(
        w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _, _, x4_tt, w4_tt = _block_inputs(device, 4, 4, True)
    _, _, _, w4_t32 = _block_inputs(device, 4, 4, False)

    cases = {
        "s1": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(
            x1_tt, w_tt, epsilon=EPS, output_layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        "s1_dram": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(
            x1_tt, w_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        "s4": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(x4_tt, w4_tt, epsilon=EPS),
        "s4_dram": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(
            x4_tt, w4_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        "s4_w32": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(x4_tt, w4_t32, epsilon=EPS),
        "s4_rm_dram": lambda: ttnn.experimental.deepseek.sharded_rmsnorm(
            x4_tt, w4_tt, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_layout=ttnn.ROW_MAJOR_LAYOUT
        ),
    }
    for name, fn in cases.items():
        ttnn.deallocate(fn())
        ttnn.synchronize_device(device)
        signpost(name)
        for _ in range(20):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(device)
