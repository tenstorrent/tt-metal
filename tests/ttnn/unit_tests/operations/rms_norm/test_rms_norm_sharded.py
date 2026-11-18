# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2 — sharded placements, consumed in place.

HEIGHT_SHARDED -> LOCAL (one line per shard core); WIDTH / BLOCK_SHARDED -> W_SPLIT (one line per shard-grid row).
TILE shards back cb_x / cb_y zero-copy; ROW_MAJOR shards are re-strided core-locally. Covers ragged shard grids
(padded last shard row / column), poisoned tile padding inside a resident shard (must be masked, never zeroed in the
caller's tensor), multi-row WIDTH grids, RM γ slices off the DRAM read grain, and the output inheriting the shard spec.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}


def _ref(x, gamma, eps=1e-6):
    xf = x.float()
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        y = y * gamma.float().reshape(-1)
    return y


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _grid(ncores_x, ncores_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores_x - 1, ncores_y - 1))})


def _row_major_cores(n, gx):
    full, rem = divmod(n, gx)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(set(ranges))


def _mc(ml, shard_shape, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(shard_shape), orientation))


def _run(device, shape, dtype, layout, mc, *, gamma_dtype=ttnn.bfloat16, gamma_layout=None, fp32_acc=True,
         fidelity=ttnn.MathFidelity.HiFi4, poison=None, pcc=None):
    torch.manual_seed(0)
    gamma_layout = layout if gamma_layout is None else gamma_layout
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    g, tg = None, None
    if gamma_dtype is not None:
        g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = ttnn.from_torch(g, dtype=gamma_dtype, layout=gamma_layout, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if poison is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, poison)
        if tg is not None and gamma_layout == ttnn.TILE_LAYOUT:
            tg = ttnn.fill_implicit_tile_padding(tg, poison)
    x_before = ttnn.to_torch(tx)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=tx.memory_config())
    assert out.memory_config() == tx.memory_config()
    assert out.dtype == dtype and out.layout == layout
    actual = ttnn.to_torch(out).float()
    assert torch.isfinite(actual).all()
    # in place means read in place: the caller's input shard is untouched
    assert torch.equal(ttnn.to_torch(tx), x_before)
    p = _pcc(_ref(x, g), actual)
    assert p >= (pcc if pcc is not None else PCC[dtype]), f"PCC {p}"


TILE_CASES = [
    # (id, shape, memory_layout, shard_shape, grid)
    ("height_8", (1, 1, 256, 512), ML.HEIGHT_SHARDED, (32, 512), _grid(8, 1)),
    ("height_multirow_shard", (1, 1, 2048, 256), ML.HEIGHT_SHARDED, (256, 256), _grid(8, 1)),
    ("height_ragged", (3, 1, 100, 128), ML.HEIGHT_SHARDED, (160, 128), _grid(3, 1)),  # 12 tile-rows: 5, 5, 2
    ("width_8x1", (1, 1, 32, 2048), ML.WIDTH_SHARDED, (32, 256), _grid(8, 1)),
    ("width_multirow_grid", (1, 1, 64, 4096), ML.WIDTH_SHARDED, (64, 64), _row_major_cores(64, 11)),
    ("width_ragged_w", (1, 1, 32, 1000), ML.WIDTH_SHARDED, (32, 192), _grid(6, 1)),  # last core: 40 valid elems
    ("block_8x8", (1, 1, 256, 512), ML.BLOCK_SHARDED, (32, 64), _grid(8, 8)),
    ("block_ragged_perf", (1, 1, 640, 6144), ML.BLOCK_SHARDED, (64, 704), _grid(9, 10)),
    ("block_ragged_rows", (1, 1, 7168, 1024), ML.BLOCK_SHARDED, (736, 128), _grid(8, 10)),
]


@pytest.mark.parametrize("case", TILE_CASES, ids=[c[0] for c in TILE_CASES])
def test_sharded_tile(device, case):
    _, shape, ml, shard_shape, grid = case
    _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, _mc(ml, shard_shape, grid))


@pytest.mark.parametrize(
    "shape, ml, shard_shape, grid",
    [
        ((1, 1, 32, 40), ML.HEIGHT_SHARDED, (32, 64), _grid(1, 1)),
        ((1, 1, 224, 72), ML.HEIGHT_SHARDED, (32, 96), _grid(7, 1)),
        ((1, 1, 32, 200), ML.WIDTH_SHARDED, (32, 32), _grid(7, 1)),  # last core: partial only (w_full = 0)
        ((1, 1, 32, 1000), ML.WIDTH_SHARDED, (32, 192), _grid(6, 1)),  # partial + ragged tail tiles
        ((1, 1, 40, 136), ML.BLOCK_SHARDED, (32, 64), _grid(3, 2)),
    ],
    ids=["height_w40", "height_w72", "width_w200", "width_w1000_ragged", "block_40x136"],
)
@pytest.mark.parametrize("fp32_acc", [True, False])
def test_sharded_tile_poisoned_padding(device, shape, ml, shard_shape, grid, fp32_acc):
    _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, _mc(ml, shard_shape, grid), fp32_acc=fp32_acc,
         poison=1000.0)


RM_CASES = [
    # (id, shape, dtype, memory_layout, shard_shape, grid)
    ("height_19sticks", (1, 1, 2048, 256), ttnn.bfloat16, ML.HEIGHT_SHARDED, (19, 256), _row_major_cores(108, 11)),
    ("height_1stick_w56", (1, 1, 17, 50), ttnn.bfloat16, ML.HEIGHT_SHARDED, (1, 56), _row_major_cores(17, 11)),
    ("width_40elem", (1, 1, 32, 4096), ttnn.bfloat16, ML.WIDTH_SHARDED, (32, 40), _row_major_cores(103, 11)),
    ("width_fp32_4elem", (1, 1, 32, 50), ttnn.float32, ML.WIDTH_SHARDED, (32, 4), _row_major_cores(13, 11)),
    ("block_100x47", (1, 1, 100, 47), ttnn.bfloat16, ML.BLOCK_SHARDED, (10, 8), _grid(6, 10)),
    ("block_aligned", (2, 1, 64, 1024), ttnn.bfloat16, ML.BLOCK_SHARDED, (16, 128), _grid(8, 8)),
]


@pytest.mark.parametrize("case", RM_CASES, ids=[c[0] for c in RM_CASES])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32, None], ids=["g_bf16", "g_fp32", "no_g"])
def test_sharded_row_major(device, case, gamma_dtype):
    _, shape, dtype, ml, shard_shape, grid = case
    _run(device, shape, dtype, ttnn.ROW_MAJOR_LAYOUT, _mc(ml, shard_shape, grid), gamma_dtype=gamma_dtype)


@pytest.mark.parametrize(
    "shape, ml, shard_shape, grid",
    [
        ((1, 1, 32, 1024), ML.WIDTH_SHARDED, (32, 128), _grid(8, 1)),
        ((1, 1, 32, 5120), ML.WIDTH_SHARDED, (32, 160), _grid(8, 4)),
        ((1, 1, 32, 7168), ML.WIDTH_SHARDED, (32, 256), _grid(7, 4)),
        ((1, 1, 7168, 2048), ML.BLOCK_SHARDED, (736, 192), _grid(11, 10)),
    ],
    ids=["w1024", "w5120", "w7168", "block_7168x2048"],
)
def test_sharded_perf_profile(device, shape, ml, shard_shape, grid):
    """The sharded perf loose-case config: bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ, PCC 0.9995."""
    _run(device, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, _mc(ml, shard_shape, grid), fp32_acc=False,
         fidelity=ttnn.MathFidelity.HiFi2, pcc=0.9995)


def test_sharded_col_major_block(device):
    grid = _grid(4, 2)
    mc = _mc(ML.BLOCK_SHARDED, (64, 256), grid, ttnn.ShardOrientation.COL_MAJOR)  # rows -> x, W -> y
    _run(device, (1, 1, 256, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT, mc)


def test_sharded_tile_rm_gamma_fp32(device):
    _run(device, (1, 1, 128, 4096), ttnn.float32, ttnn.TILE_LAYOUT,
         _mc(ML.BLOCK_SHARDED, (32, 512), _grid(8, 4)), gamma_dtype=ttnn.bfloat16,
         gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


def test_sharded_default_output_inherits(device):
    torch.manual_seed(0)
    x = torch.randn((1, 1, 64, 256), dtype=torch.bfloat16)
    mc = _mc(ML.WIDTH_SHARDED, (64, 64), _grid(4, 1))
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    out = rms_norm(tx)
    assert out.memory_config() == mc
    assert _pcc(_ref(x, None), ttnn.to_torch(out).float()) >= 0.995


WIDE_HEIGHT_CASES = [
    # A HEIGHT shard pins a whole row to one core: γ (and RM staging) must stream in W-chunks next to the shards.
    # (id, shape, dtype, layout, gamma_dtype, gamma_layout, shard_shape, grid)
    ("tile_bf16_w8192_g_fp32", (1, 1, 32, 8192), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.float32, ttnn.TILE_LAYOUT,
     (32, 8192), _grid(1, 1)),
    ("tile_fp32_w4096_g_fp32_rm", (1, 1, 128, 4096), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.float32,
     ttnn.ROW_MAJOR_LAYOUT, (32, 4096), _grid(4, 1)),
    ("tile_bf16_w11008", (1, 1, 160, 11008), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.TILE_LAYOUT,
     (32, 11008), _grid(5, 1)),
    ("rm_bf16_w3072", (1, 1, 224, 3072), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
     (3, 3072), _row_major_cores(75, 11)),
    ("rm_bf16_w11008", (1, 1, 160, 11008), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16,
     ttnn.ROW_MAJOR_LAYOUT, (2, 11008), _row_major_cores(80, 11)),
    ("rm_fp32_w8192", (1, 1, 128, 8192), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT,
     (2, 8192), _row_major_cores(64, 11)),
    ("rm_bf16_w4095_multirow", (1, 1, 96, 4095), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32,
     ttnn.ROW_MAJOR_LAYOUT, (48, 4096), _grid(2, 1)),
]


@pytest.mark.parametrize("case", WIDE_HEIGHT_CASES, ids=[c[0] for c in WIDE_HEIGHT_CASES])
def test_sharded_wide_height(device, case):
    _, shape, dtype, layout, gd, gl, shard_shape, grid = case
    _run(device, shape, dtype, layout, _mc(ML.HEIGHT_SHARDED, shard_shape, grid), gamma_dtype=gd, gamma_layout=gl)
