# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — causal block-skip perf check.

DO NOT DELETE.

The causal path block-skips whole future KV chunks (Q chunk qc processes only
chunks j with skv_off < sq_off + sq_valid, i.e. ~(qc+1) of n_kv_chunks), so on a
square S×S causal self-attention it does ≈ half the per-core KV work of an
equivalent full-mask CUSTOM run (which streams + processes ALL n_kv_chunks and
masks the future with an additive triangular tensor). This file measures that:

  1. Correctness gate (plain run): causal and custom(triangular) both match the
     torch causal reference — proving the two are numerically equivalent, so any
     device-ns delta between them is attributable to the block-skip alone.
  2. Perf comparison (`--profile`): the test loops causal N times then custom N
     times IN THE SAME PROCESS (shared AICLK — the changelog documents ~1.8x clock
     drift between fresh invocations, invalid for A/B; same-process back-to-back is
     valid). Read the ops_perf CSV: the causal generic_op DEVICE KERNEL DURATION
     rows are shorter than the custom rows.

Measure:
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_causal_perf.py::test_causal_blockskip_vs_fullmask
"""

from __future__ import annotations

import math

import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Square self-attention with high B·H and modest n_q_chunks: each core's contiguous
# work-unit range spans the full qc distribution (0..n_q-1), so the per-core KV work
# averages to ≈ (n_q+1)/(2·n_q) of the full-mask cost — the block-skip shows up on the
# critical path (with naive contiguous assignment; the full ~2x needs causal
# load-balancing, a future perf refinement — FlashAttention.md §causal load-balancing).
# S=1024 -> Skv_t=32, chunk 8 -> n_kv=4, n_q=4; B·H=128 work-unit blocks spread qc evenly.
PERF_SHAPE = (16, 8, 1024, 64)
ITERS = 6  # 1 warm-up (JIT) + 5 measured rows per variant


def test_causal_blockskip_vs_fullmask(device):
    torch.manual_seed(0)
    B, H, S, D = PERF_SHAPE
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)

    # Reference: torch causal SDPA. The custom variant uses the identical triangular
    # additive mask, so it computes the SAME result (verified) — the ONLY difference
    # vs causal is that custom processes every KV chunk (no block-skip).
    ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), is_causal=True, scale=scale)
    tri = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    tri.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tm = ttnn.from_torch(tri, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Causal (block-skip) — first N rows in the ops_perf CSV.
    out_causal = None
    for _ in range(ITERS):
        out_causal = scaled_dot_product_attention(tq, tk, tv, is_causal=True, scale=scale)

    # Custom full-mask (no block-skip) — next N rows in the CSV.
    out_custom = None
    for _ in range(ITERS):
        out_custom = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, scale=scale)

    # Both must match the causal reference (equivalent math → device-ns delta is the
    # block-skip alone).
    assert_with_pcc(ref, ttnn.to_torch(out_causal).to(torch.float32), pcc=0.995)
    assert_with_pcc(ref, ttnn.to_torch(out_custom).to(torch.float32), pcc=0.995)
    logger.info(
        f"SDPA causal block-skip check {PERF_SHAPE}: causal & custom both match causal ref "
        f"(PCC>=0.995). Under --profile the causal generic_op rows are shorter than custom."
    )
