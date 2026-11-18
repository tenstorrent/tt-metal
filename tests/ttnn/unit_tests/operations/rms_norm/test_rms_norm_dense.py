# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 — dense rinv (W_SPLIT lines with tall blocks).

Σx² of up to 32 tile-rows is packed into one dense tile by a matmul against one-hot column selectors; the gather,
the root's combine + finalize and the rinv multicast run on ⌈rows/32⌉ tiles; every core expands rinv back with the
transposed selectors. Each case asserts select_blocking() really chose the dense schedule, then checks numerics:
TILE / ROW_MAJOR, bf16 / fp32 / bfloat8_b, fp32 DEST on / off, with / without γ (TILE and RM γ), ragged shard
grids, and blocks of more than 32 tile-rows (two dense groups per block).
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

ML = ttnn.TensorMemoryLayout
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


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


def _grid(nx, ny):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))})


@pytest.fixture
def blocking_log(monkeypatch):
    log = []
    orig = pd.select_blocking

    def wrapped(*args, **kwargs):
        b = orig(*args, **kwargs)
        log.append(b)
        return b

    monkeypatch.setattr(pd, "select_blocking", wrapped)
    return log


def _run(device, blocking_log, shape, dtype, layout, ml, shard_shape, grid, *, gamma_dtype=ttnn.bfloat16,
         gamma_layout=None, fp32_acc=True, fidelity=ttnn.MathFidelity.HiFi4, pcc=0.999, min_groups=1):
    torch.manual_seed(0)
    gamma_layout = layout if gamma_layout is None else gamma_layout
    mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR))
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    x = ttnn.to_torch(tx)  # bfloat8_b: the reference sees the quantized input
    g, tg = None, None
    if gamma_dtype is not None:
        g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = ttnn.from_torch(g, dtype=gamma_dtype, layout=gamma_layout, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.to_torch(tg).reshape(1, 1, 1, -1)[..., : shape[-1]]
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
    assert blocking_log and blocking_log[-1].dense, f"dense schedule not selected: {blocking_log[-1:]}"
    assert pd._dense_groups(blocking_log[-1].block_rows) >= min_groups, blocking_log[-1]
    actual = ttnn.to_torch(out).float()
    assert torch.isfinite(actual).all()
    p = _pcc(_ref(x, g), actual)
    assert p >= pcc, f"PCC {p}"


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["acc", "noacc"])
@pytest.mark.parametrize(
    "shape, shard_shape, grid, min_groups",
    [
        ((1, 1, 7168, 2048), (736, 192), _grid(11, 10), 1),  # perf case: 23 tile-rows / core, ragged grid
        ((1, 1, 8192, 1024), (1024, 128), _grid(8, 8), 1),  # 32 tile-rows: one full dense group
        ((1, 1, 3072, 512), (1536, 128), _grid(4, 2), 2),  # 48 tile-rows: two dense groups (32 + 16)
    ],
    ids=["7168x2048_11x10", "8192x1024_8x8", "3072x512_two_groups"],
)
def test_dense_block_tile(device, blocking_log, shape, shard_shape, grid, min_groups, fp32_acc):
    _run(device, blocking_log, shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, shard_shape, grid,
         fp32_acc=fp32_acc, min_groups=min_groups, pcc=0.9995)


def test_dense_perf_config(device, blocking_log):
    """The perf-profile config (HiFi2, fp32 DEST off) on a dense perf case keeps its 0.9995 gate."""
    _run(device, blocking_log, (1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (1024, 128),
         _grid(8, 8), fp32_acc=False, fidelity=ttnn.MathFidelity.HiFi2, pcc=0.9995)


@pytest.mark.parametrize("gamma", ["none", "tile_bf16", "rm_bf16"])
def test_dense_gamma_variants(device, blocking_log, gamma):
    gamma_dtype = None if gamma == "none" else ttnn.bfloat16
    gamma_layout = ttnn.ROW_MAJOR_LAYOUT if gamma == "rm_bf16" else ttnn.TILE_LAYOUT
    _run(device, blocking_log, (1, 1, 2048, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (256, 128),
         _grid(8, 8), gamma_dtype=gamma_dtype, gamma_layout=gamma_layout)


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["acc", "noacc"])
def test_dense_fp32_input(device, blocking_log, fp32_acc):
    if not fp32_acc:
        pytest.skip("fp32 input with fp32_dest_acc_en=False is a permanent exclusion")
    _run(device, blocking_log, (1, 1, 2048, 512), ttnn.float32, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (256, 64),
         _grid(8, 8), gamma_dtype=ttnn.float32, fp32_acc=fp32_acc, pcc=0.9999)


def test_dense_bfloat8_b(device, blocking_log):
    _run(device, blocking_log, (1, 1, 2048, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (256, 128),
         _grid(8, 8), fp32_acc=False, pcc=0.995)


@pytest.mark.parametrize("fp32_acc", [True, False], ids=["acc", "noacc"])
def test_dense_block_row_major(device, blocking_log, fp32_acc):
    # 128 sticks = 4 tile-rows per core: the smallest dense block (DENSE_MIN_ROWS).
    _run(device, blocking_log, (1, 1, 1024, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ML.BLOCK_SHARDED,
         (128, 128), _grid(8, 8), fp32_acc=fp32_acc, pcc=0.999)


def test_dense_width_sharded_multirow(device, blocking_log):
    _run(device, blocking_log, (1, 1, 256, 2048), ttnn.bfloat16, ttnn.TILE_LAYOUT, ML.WIDTH_SHARDED, (256, 256),
         _grid(8, 1))
