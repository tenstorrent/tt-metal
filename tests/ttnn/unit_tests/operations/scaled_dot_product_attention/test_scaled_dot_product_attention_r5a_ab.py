# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R5a ablation: bound the PV-matmul + O-rescale/accumulate headroom on the flagged shape.

R5a proposes batching consecutive KV chunks' P·V into one wider-K matmul. K-batching keeps
the total FMA but amortizes the per-call/pack overhead AND the per-chunk O rescale+accumulate
over B chunks. This measures the CEILING of that win via /perf-measure ablation — stub the PV
payload, keep every CB reserve/wait/pop/push — same-session (steady AICLK) so the deltas are
valid.

  SDPA_ABLATE_PV unset -> baseline (normal PV + rescale + accumulate)
  SDPA_ABLATE_PV=1     -> PV matmul FMA+pack stubbed  (delta = PV-matmul cost)
  SDPA_ABLATE_PV=2     -> PV matmul + rescale + accumulate stubbed (delta = whole PV+accum zone)
  SDPA_ABLATE_PV=3     -> QK^T + PV + rescale + accumulate stubbed (delta = total matmul+accum)

Measure (rows in the ops CSV are ordered baseline, ablate1, ablate2, each warmup+N):
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_r5a_ab.py
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


def test_sdpa_r5a_ablation(device):
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

    def run_variant(label, ablate):
        if ablate is None:
            os.environ.pop("SDPA_ABLATE_PV", None)
        else:
            os.environ["SDPA_ABLATE_PV"] = str(ablate)
        out = None
        for _ in range(ITERS):
            out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        ttnn.to_torch(out)  # force completion
        logger.info(f"R5a ablation [{label}] done")

    run_variant("baseline", None)
    run_variant("stub_pv_matmul", 1)
    run_variant("stub_pv_matmul_and_accum", 2)
    run_variant("stub_both_matmuls_and_accum", 3)
    os.environ.pop("SDPA_ABLATE_PV", None)
