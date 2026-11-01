# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 perf probe (not collected by a directory run — no `test_` prefix).

    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/rms_norm/perf_r3_rms_norm.py

One dispatch per cell at the perf-profile config (bf16, fp32_dest_acc_en=False, HiFi2, TILE x, TILE gamma);
prints the chosen blocking so the CSV rows can be matched to knob settings.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

SHAPES = [
    (1, 1, 640, 6144),
    (1, 1, 640, 7168),
    (1, 1, 1024, 1024),
    (1, 1, 8192, 7168),
    (1, 1, 7168, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 1024),
    (1, 1, 32, 7168),
    # guard set (other kernel paths / regimes)
    (1, 1, 2048, 256),
    (4, 1, 512, 512),
    (1, 1, 32, 1024),
    (1, 1, 32, 16384),
    (1, 1, 352, 4096),
]
_sel = os.environ.get("R3_SHAPES")
if _sel:
    SHAPES = [s for s in SHAPES if "x".join(map(str, s)) in _sel.split(",")]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_perf(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    is_rm = os.environ.get("R3_RM") == "1"
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
        f"\nR3 {shape}: k={blk.k} groups={len(blk.groups)} cores={ncores} b={blk.block_row_tiles} "
        f"d={blk.x_block_depth} rows/core={blk.groups[0][0].core_rows} fp={blk.footprint}"
    )
    for _ in range(int(os.environ.get("R3_REPS", "1")) - 1):
        rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    y = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float()
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"R3 {shape}: pcc={pcc:.6f}")
    assert pcc >= 0.9995, pcc
