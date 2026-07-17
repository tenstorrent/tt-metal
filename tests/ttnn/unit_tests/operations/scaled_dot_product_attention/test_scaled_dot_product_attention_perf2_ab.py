# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 2 ablation: split the compute-bound residual into PAYLOAD vs per-phase OVERHEAD.

Prior refinements (R5a/R5b/Perf1) established the flagged 1x10x9472x128 shape is
compute-bound: DM 2.1% (hidden), matmul+accum 7.3%, residual ~92.7% "softmax + overhead".
They never freshly ablated WHERE that 92.7% goes. This measures it same-session (steady
AICLK) via two measurement-only compile-time gates:

  SDPA_ABLATE_PV=3                        -> stub both matmuls + O rescale/accumulate
  SDPA_ABLATE_PV=3 + SDPA_ABLATE_SOFTMAX=1 -> ALSO stub the row-max reduce + exp dual-pack

The last variant keeps EVERY CB reserve/wait/pop/push (so per-phase fill-drain/init/reconfig/
CB-sync overhead is fully paid) but zeroes the matmul FMA, the row-max reduce math, and the
SFPU exp. Its duration is the PURE-OVERHEAD FLOOR. Then:
  payload  = baseline - floor   (matmul+accum+reduce+exp math)
  overhead = floor              (per-phase fill/drain + init + reconfig + CB-sync + tiny phases)

If overhead dominates -> only coarser blocks help (L1/divisor-blocked -> dead).
If payload dominates  -> attacking exp/reduce could help.

Measure:
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_perf2_ab.py
"""

from __future__ import annotations

import math
import os

import torch
import ttnn
from loguru import logger

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PERF_SHAPE = (1, 10, 9472, 128)
ITERS = 6  # 1 warm-up + 5 measured per variant


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def test_sdpa_perf2_ablation(device):
    torch.manual_seed(1234)
    B, H, S, D = PERF_SHAPE
    Q, K, V = _fa_rand(B, H, S, D), _fa_rand(B, H, S, D), _fa_rand(B, H, S, D)
    scale = 1.0 / math.sqrt(D)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    def run_variant(label, pv, softmax):
        for k, v in (("SDPA_ABLATE_PV", pv), ("SDPA_ABLATE_SOFTMAX", softmax)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        out = None
        for _ in range(ITERS):
            out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        ttnn.to_torch(out)  # force completion
        logger.info(f"Perf2 ablation [{label}] done")

    run_variant("baseline", None, None)
    run_variant("stub_matmuls+accum", 3, None)
    run_variant("stub_matmuls+accum+softmax(overhead_floor)", 3, 1)
    os.environ.pop("SDPA_ABLATE_PV", None)
    os.environ.pop("SDPA_ABLATE_SOFTMAX", None)
