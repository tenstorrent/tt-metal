# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R5 same-session A/B perf measurement for the flagged shape.

Measures the R5 compute-side lever — growing the PV matmul output-subblock HEIGHT to
fill the DEST budget (decomp_h). Runs the flagged 1x10x9472x128 shape (bf16,
fp32_dest_acc_en=False) N times with the lever ON (SDPA_PV_SB_H unset -> grow_subblock_h
= 1 -> PV out_subblock_h = 2), then N times with it OFF (SDPA_PV_SB_H=0 forces
out_subblock_h = 1, the pre-R5 baseline) — all in ONE process so the AICLK is steady
(fresh-invocation drift is ~1.8x and would invalidate a cross-session ns A/B). Two
distinct programs (grow_subblock_h flips a CT arg), each warmed once.

Measure (rows in the ops CSV are ordered: warmup+N lever-ON, then warmup+N lever-OFF):
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_r5_ab.py
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


def test_sdpa_r5_ab(device):
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
        logger.info(f"R5 A/B [{label}]: PCC>=0.997 held")

    os.environ["SDPA_PV_SB_H"] = "1"  # lever ON -> PV out_subblock_h grown to fill DEST
    run_variant("PV_SB_H_ON")
    os.environ.pop("SDPA_PV_SB_H", None)  # lever OFF (parked default) -> out_subblock_h=1
    run_variant("PV_SB_H_OFF")
