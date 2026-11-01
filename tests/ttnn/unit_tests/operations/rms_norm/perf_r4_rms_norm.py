# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4 perf probe — HEIGHT_SHARDED (not collected by a directory run — no `test_` prefix).

    R4_GAMMA_MCAST=0|1 scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/rms_norm/perf_r4_rms_norm.py

One dispatch per cell at the perf-profile config (bf16, fp32_dest_acc_en=False, HiFi2, gamma in the input's
Layout), auto HEIGHT shard spec. R4_GAMMA_MCAST toggles pd.GAMMA_MCAST_SHARDED_K1 (unset: the module default).
"""

import os

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

SHAPES = [
    (1, 1, 4096, 4096),
    (1, 1, 8192, 1024),
    (1, 1, 4096, 7168),
    (7136, 736),
    (99991, 64),
]
_sel = os.environ.get("R4_SHAPES")
if _sel:
    SHAPES = [s for s in SHAPES if "x".join(map(str, s)) in _sel.split(",")]


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_perf(device, shape, layout, monkeypatch):
    if "R4_GAMMA_MCAST" in os.environ:
        monkeypatch.setattr(pd, "GAMMA_MCAST_SHARDED_K1", os.environ["R4_GAMMA_MCAST"] == "1")
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    mc = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=layout, dtype=ttnn.bfloat16, device=device
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=layout, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    print(f"\nR4 {shape} {layout}: shard={mc.shard_spec.shape} cores={mc.shard_spec.grid.num_cores()}")
    y = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)).float()
    xf = x.float()
    ref = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g.float()
    pcc = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
    print(f"R4 {shape}: pcc={pcc:.6f}")
    assert pcc >= 0.9995, pcc
