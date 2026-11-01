# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5 perf probe (not collected by a directory run — no `test_` prefix).

    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/rms_norm/perf_r5_rms_norm.py

Few-tile-row wide-W INTERLEAVED cells at the perf-profile config (bf16, fp32_dest_acc_en=False, HiFi2, TILE x,
TILE gamma). One dispatch per (shape, forced k) so the CSV rows map 1:1 to the printed blocking.
Env:
  R5_SHAPES  comma list of AxBxCxD ids to keep
  R5_KS      comma list of forced w_split_cores values (sweep; default: the op's own choice only)
  R5_RM=1    ROW_MAJOR input
  R5_NOGAMMA=1  no gamma (ablation)       R5_NOMCAST=1  per-core gamma load (GAMMA_MCAST off)
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

SHAPES = [
    (1, 1, 32, 32768),
    (1, 1, 32, 16384),
    (1, 1, 160, 11008),
    (1, 1, 96, 6144),
    (1, 1, 32, 7168),
    (1, 1, 32, 5120),
    # guard set (regimes R5 must not move: R3 mid tile-rows, k = 1 prefill)
    (1, 1, 640, 6144),
    (1, 1, 1024, 1024),
    (1, 1, 32, 1024),
    (1, 1, 352, 4096),
]
_sel = os.environ.get("R5_SHAPES")
if _sel:
    SHAPES = [s for s in SHAPES if "x".join(map(str, s)) in _sel.split(",")]
KS = [int(k) for k in os.environ.get("R5_KS", "0").split(",")]


@pytest.mark.parametrize("k_force", KS, ids=lambda k: f"k{k}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_perf(device, shape, k_force, monkeypatch):
    if k_force:
        monkeypatch.setattr(pd, "_choose_k", lambda *a, **kw: k_force)
    if os.environ.get("R5_NOMCAST") == "1":
        monkeypatch.setattr(pd, "GAMMA_MCAST", False)
    use_gamma = os.environ.get("R5_NOGAMMA") != "1"
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    is_rm = os.environ.get("R5_RM") == "1"
    layout = ttnn.ROW_MAJOR_LAYOUT if is_rm else ttnn.TILE_LAYOUT
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    geo = pd.derive_geometry(tx.shape)
    sz = pd._Sizes(tx.dtype, tg.dtype, is_rm, True, False)
    grid = device.compute_with_storage_grid_size()
    blk = pd.select_blocking(geo, sz, grid.x, grid.y, pd.cb_l1_budget(device), 2)
    ncores = sum(len(gr) for gr in blk.groups)
    print(
        f"\nR5 {shape} kf={k_force}: k={blk.k} slice={blk.slice_w_tiles} groups={len(blk.groups)} cores={ncores} "
        f"owner={blk.opts.owner_combine} b={blk.block_row_tiles} d={blk.x_block_depth} "
        f"rows/core={blk.groups[0][0].core_rows} gamma_mcast={pd._gamma_mcast_groups(blk, grid.x) is not None}"
    )
    y = ttnn.to_torch(rms_norm(tx, gamma=tg if use_gamma else None, compute_kernel_config=cfg)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * (g.float() if use_gamma else 1.0)
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"R5 {shape} kf={k_force}: pcc={pcc:.6f}")
    assert pcc >= 0.9995, pcc


# Sharded guard (WIDTH / BLOCK k > 1 share the writer's exchange + gamma path). R5_SHARDED=1 runs them;
# R5_SHARDED_SEL = comma list of indices into PINNED.
PINNED = [
    (32, 1024, "W", [32, 128], (8, 1)),
    (32, 2304, "W", [32, 256], (9, 1)),
    (32, 5120, "W", [32, 160], (8, 4)),
    (32, 7168, "W", [32, 256], (7, 4)),
    (640, 6144, "B", [64, 704], (9, 10)),
    (640, 7168, "B", [64, 1024], (7, 10)),
    (7168, 1024, "B", [736, 128], (8, 10)),
    (8192, 1024, "B", [1024, 128], (8, 8)),
]
_ssel = os.environ.get("R5_SHARDED_SEL")
if _ssel:
    PINNED = [PINNED[int(i)] for i in _ssel.split(",")]


@pytest.mark.skipif(os.environ.get("R5_SHARDED") != "1", reason="R5_SHARDED=1 selects the sharded guard")
@pytest.mark.parametrize("case", PINNED, ids=lambda c: f"{c[2]}{c[0]}x{c[1]}")
def test_perf_sharded(device, case):
    from eval.sharding import shard_config

    rows, W, ml, shard, grid = case
    ml = ttnn.TensorMemoryLayout.WIDTH_SHARDED if ml == "W" else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    torch.manual_seed(0)
    x = torch.randn(1, 1, rows, W).bfloat16()
    g = torch.randn(W).bfloat16()
    mc = shard_config(shard, grid, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    y = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float()
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    assert pcc >= 0.9995, pcc
