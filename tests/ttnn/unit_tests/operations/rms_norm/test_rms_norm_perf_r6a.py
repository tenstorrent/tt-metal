# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6a — sharded cross-core round-batching + gap-aware mcast.

Two levers on the shared cross-core (`_assemble_xcore_kernels`) transport, both correct
and non-regressing:

  Lever 1 (round-batching): one cross-core round now exchanges C tile-rows' partials
    (compute produces C local partials, the writer gathers K*C, the master folds C rstds,
    broadcasts C), so BLOCK's sync rounds drop from HT_LOCAL to ceil(HT_LOCAL/C). C is a
    host tunable (`STAT_BATCH_ROWS`, L1-gated per program). C=1 is byte-identical to R4.
  Lever 2 (gap-aware mcast): a group straddling the Blackhole DRAM columns (virtual x=8,9)
    now mcasts its 1/RMS broadcast in up to TWO contiguous virtual-x runs instead of K-1
    unicast writes — unblocking the 8-wide WIDTH/BLOCK groups R6's strict-rectangle mcast
    could not reach.

Measured (blackhole_p150b, median of 8 fresh trials, exact perf config
bf16 / fp32_dest_acc_en=False / TILE / TILE gamma / HiFi2):
  BLOCK 8x8 (HT_LOCAL=32): 147729 -> 118969 ns  (5.76x -> 4.64x above achievable; lever 1)
  WIDTH 8x4 (K=32):         10204 ->   8884 ns  (1.94x -> 1.69x; lever 2 mcast vs 10173 unicast)
  WIDTH 7x4 (K=28):          9870 ->   9764 ns  (mcast vs 11173 unicast, confirms R6)
  WIDTH 8x1/9x1 (K=8/9,HT=1): flat (near achievable; single-round-latency bound)

This is NOT the golden suite — it produces device-ns numbers and locks down the batched-
round correctness (soft PCC gate 0.9995), including the short-last-round edge case where
C does not divide HT_LOCAL.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import shard_config
from ttnn.operations.rms_norm import rms_norm

_ML = ttnn.TensorMemoryLayout


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item()) / denom


def _perf_cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _ref(x, g):
    e = x.to(torch.float32)
    e = e * torch.rsqrt(e.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return e * g.to(torch.float32).reshape(-1)


# (rows, W, ml, (shard_h, shard_w), (gx, gy), HT_LOCAL) — the batching / gap-aware-mcast
# targets plus HT_LOCAL values that hit the short-last-round edge (32 % C != 0 for C in
# {2,4,8}: HT=3 -> rounds ceil(3/2)=2 short, HT=5, HT=32). gamma present + absent.
_CASES = [
    # BLOCK: HT_LOCAL = shard_h/32 -> batching engages (C = min(HT, 8, L1))
    (8192, 1024, _ML.BLOCK_SHARDED, (1024, 128), (8, 8), 32),  # HT=32, C=8 -> 4 rounds (perf target)
    (96, 512, _ML.BLOCK_SHARDED, (96, 128), (4, 3), 3),  # HT=3, C=3 -> 1 round (short: 3<8)
    (160, 512, _ML.BLOCK_SHARDED, (160, 128), (4, 4), 5),  # HT=5, C=5 -> 1 round (5<8)
    (416, 256, _ML.BLOCK_SHARDED, (416, 64), (4, 4), 13),  # HT=13, C=8 -> 2 rounds (13%8=5 short)
    # WIDTH: HT_LOCAL=1 (C=1, no batching) — gap-aware mcast on the 8-wide K
    (32, 1024, _ML.WIDTH_SHARDED, (32, 128), (8, 1), 1),  # K=8 straddles gap
    (32, 5120, _ML.WIDTH_SHARDED, (32, 160), (8, 4), 1),  # K=32 straddles gap (lever-2 win)
    (32, 7168, _ML.WIDTH_SHARDED, (32, 256), (7, 4), 1),  # K=28 gap-free (R6 mcast)
]


@pytest.mark.parametrize("gamma_on", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "rows,W,ml,shard,grid,ht",
    _CASES,
    ids=[f"{ml.name.split('_')[0].lower()}_{r}x{w}_ht{ht}" for (r, w, ml, _s, _g, ht) in _CASES],
)
def test_batched_round_correct(rows, W, ml, shard, grid, ht, gamma_on, device):
    """Batched cross-core round stays correct across HT_LOCAL (incl. short last round)."""
    torch.manual_seed(0)
    shape = (1, 1, rows, W)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    tg = torch.randn(W, dtype=torch.bfloat16)
    expected = _ref(ti, tg)

    cfg = shard_config(list(shard), grid, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    gt = None
    if gamma_on:
        gt = ttnn.from_torch(
            tg.reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=_perf_cfg(), memory_config=cfg)
    result = ttnn.to_torch(out)
    ref = (
        expected
        if gamma_on
        else (ti.to(torch.float32) * torch.rsqrt(ti.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6))
    )
    pcc = _pcc(result, ref)
    print(f"\nR6A {ml.name} ({rows},{W}) ht={ht} gamma={gamma_on} PCC={pcc:.6f}")
    assert pcc >= 0.9995, f"soft PCC gate: {pcc} < 0.9995 for {shape} ht={ht}"
