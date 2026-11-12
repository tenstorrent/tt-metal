# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rms_norm debug tests — DO NOT DELETE (documents the debugging process).

1. RM width sweep: ROW_MAJOR rows of W = 4096 / 8192 failed (pcc ~0.5) while TILE passed at the same
   widths. Sweep W to find where the RM path (tilize<c> / untilize<c> / chunked stick I/O) breaks, and
   print the block knobs (r, c, K) the host chose for each W.
2. Monotonic per-column input: out[r, col] should be proportional to x[r, col]; any column permutation
   introduced by chunked tilize/untilize shows up as a ratio != const.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


def _knobs(x, gamma, out):
    g = pd._Geometry(x, gamma, out)
    grid = x.device().compute_with_storage_grid_size()
    *_, n1, n2 = ttnn.split_work_to_cores(grid, g.tensor_rows_tiles, row_wise=True)
    mv = ttnn.device.get_memory_view(x.device(), ttnn.BufferType.L1)
    budget = mv.largest_contiguous_bytes_free_per_bank - pd.L1_SAFETY_MARGIN
    return pd._select_blocks(g, g.tensor_w_tiles, max(n1, n2), budget)


def _ref(x, eps=1e-6):
    xf = x.float()
    return xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + eps)


@pytest.mark.parametrize("W", [256, 512, 1024, 2048, 4096])
def test_rm_width_sweep(device, W):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, W, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(tx)
    print(f"W={W} knobs(r,c,K)={_knobs(tx, None, out)}")
    got = ttnn.to_torch(out).float()
    exp = _ref(x)
    err = (got - exp).abs()
    bad_cols = (err.amax(dim=-2).flatten() > 0.1).nonzero().flatten()
    print(f"W={W} max_err={err.max().item():.4f} first bad cols={bad_cols[:16].tolist()} n_bad={bad_cols.numel()}")
    assert err.max().item() < 0.1


def test_rm_monotonic_columns(device):
    """x[r, c] = 1 + c/W — every column distinct; ratio out/x must be constant per row."""
    W = 4096
    col = 1.0 + torch.arange(W, dtype=torch.float32) / W
    x = col.repeat(32, 1).reshape(1, 1, 32, W).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tx)).float()
    ratio = out / x.float()
    print("ratio row0 sample:", ratio[0, 0, 0, ::256].tolist())
    print("ratio col-spread per row0:", (ratio[0, 0, 0].max() - ratio[0, 0, 0].min()).item())
    assert torch.allclose(out, _ref(x), atol=0.05, rtol=0.05)


def test_debug_two_tile_rows_one_block(device):
    """Refinement 1 debug: block_rows=2 on one core (64-row HEIGHT shard, W=512).

    Tile-row 0 is all 1.0 (sumsq = 512, rstd = 1), tile-row 1 is all 2.0 (sumsq = 2048, rstd = 0.5):
    output must be all ones. Found: tile-row 1 of a multi-row block got a wrong rstd (Phase 0 never
    put more than one tile-row on a core in its tests).
    """
    from eval.sharding import shard_config

    x = torch.ones(1, 1, 64, 512)
    x[..., 32:, :] = 2.0
    mc = shard_config(
        [64, 512], (1, 1), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32
    )
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    out = ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg)).float()
    assert torch.allclose(
        out, torch.ones_like(out), atol=1e-2
    ), f"tile-row maxes {out[0,0].reshape(2,32,512).amax((1,2))}"
