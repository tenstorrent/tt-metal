# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R3e same-session A/B perf measurement for the flagged shape.

Runs the flagged 1x10x9472x128 shape (bf16, fp32_dest_acc_en=False) N times with
the R3e L1-accumulate-during-exp row-sum fusion ON, then N times with it OFF
(SDPA_FUSE_ROWSUM=0 forces the pre-R3e reduce<SUM> path in the SAME regime) — all
in ONE process so the AICLK is steady (fresh-invocation drift is ~1.8x and would
invalidate a cross-session ns A/B). Two distinct programs (fuse_rowsum flips a CT
arg), each warmed once.

Measure (rows in the ops CSV are ordered: warmup+N fused, then warmup+N reduce):
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_r3e_ab.py
"""

from __future__ import annotations

import math
import os

import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PERF_SHAPE = (1, 10, 9472, 128)
ITERS = 6  # 1 warm-up + 5 measured per variant


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def test_sdpa_r3e_ab(device):
    torch.manual_seed(1234)
    B, H, S, D = PERF_SHAPE
    Q, K, V = _fa_rand(B, H, S, D), _fa_rand(B, H, S, D), _fa_rand(B, H, S, D)
    scale = 1.0 / math.sqrt(D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), attn_mask=None, is_causal=False, scale=scale
    )
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    def run_variant(label):
        out = None
        for _ in range(ITERS):
            out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        result = ttnn.to_torch(out).to(torch.float32)
        assert_with_pcc(expected, result, pcc=0.997)
        logger.info(f"R3e A/B [{label}]: PCC>=0.997 held")

    os.environ["SDPA_FUSE_ROWSUM"] = "1"  # fused ON (default gate for this regime)
    run_variant("FUSED_ON")
    os.environ["SDPA_FUSE_ROWSUM"] = "0"  # fused OFF -> reduce<SUM> path, same regime
    run_variant("FUSE_OFF_REDUCE")
    os.environ.pop("SDPA_FUSE_ROWSUM", None)
