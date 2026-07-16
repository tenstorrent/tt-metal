# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for the R3 data-movement refinement.

Runs the mandatory perf-flagged shape (feature_spec.LOOSE_CASES):
  1 x 10 x 9472 x 128, bf16, MHA self-attn, fp32_dest_acc_en=False, HiFi2.

The test loops the op N times so a `--profile` run yields N device-duration
rows for the SDPA generic_op in the ops_perf CSV; take the median of the
warm rows (drop the first) for a defensible device-ns number. Correctness
(PCC >= 0.997, the soft golden gate) is asserted on the last iteration.

Run correctness (plain):
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_perf.py

Measure device-ns (profiled — masks exit code, measurement only):
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_perf.py
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Flagged shape + regime (feature_spec.LOOSE_CASES).
PERF_SHAPE = (1, 10, 9472, 128)
PERF_ITERS = 6  # 1 warm-up (JIT compile) + 5 measured rows


def _fa_rand(*shape):
    """Flash-attention heavy-tailed distribution (matches golden helpers.fa_rand)."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def test_sdpa_perf_flagged_shape(device):
    torch.manual_seed(1234)
    B, H, S, D = PERF_SHAPE
    Q = _fa_rand(B, H, S, D)
    K = _fa_rand(B, H, S, D)
    V = _fa_rand(B, H, S, D)

    scale = 1.0 / math.sqrt(D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), attn_mask=None, is_causal=False, scale=scale
    )

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )

    out = None
    for _ in range(PERF_ITERS):
        out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)

    result = ttnn.to_torch(out).to(torch.float32)
    pcc_ok, pcc_msg = 0.997, ""
    assert_with_pcc(expected, result, pcc=pcc_ok)
    logger.info(f"SDPA perf shape {PERF_SHAPE}: PCC gate {pcc_ok} passed")


# R3 guard set: confirm the batched reader/writer holds across the config-spanning
# set the "Done when" names. The golden suite already covers mask none/custom ×
# small/large with DRAM output; this fills the L1-output-placement half (both mask
# modes, small + medium shape) that the golden suite does not exercise.
@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (1, 4, 512, 64)], ids=["small", "medium"])
@pytest.mark.parametrize("mask_mode", ["none", "custom"])
@pytest.mark.parametrize("mem", ["dram", "l1"], ids=["dram", "l1"])
def test_sdpa_guard_set(device, shape, mask_mode, mem):
    torch.manual_seed(0)
    B, H, S, D = shape
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)

    torch_mask = None
    if mask_mode == "custom":
        torch_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
        torch_mask.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))

    scale = 1.0 / math.sqrt(D)
    expected = torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V.float(),
        attn_mask=torch_mask.float() if torch_mask is not None else None,
        is_causal=False,
        scale=scale,
    )

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tm = (
        ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_mask is not None
        else None
    )
    mem_cfg = ttnn.L1_MEMORY_CONFIG if mem == "l1" else ttnn.DRAM_MEMORY_CONFIG

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, scale=scale, memory_config=mem_cfg)
    result = ttnn.to_torch(out).to(torch.float32)
    assert_with_pcc(expected, result, pcc=0.99)
    logger.info(f"SDPA guard {shape} mask={mask_mode} mem={mem}: PCC>=0.99 passed")
