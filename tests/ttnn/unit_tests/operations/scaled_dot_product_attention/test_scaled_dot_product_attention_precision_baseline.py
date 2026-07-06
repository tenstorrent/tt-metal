# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Flash Attention).

Measures PCC, max abs error, mean abs error, and relative RMS error across
3 shapes (small, medium, one larger) that fit in the single-block path
(S ≤ 32 — the only path that currently works without hanging).

Uses assert_with_pcc from tests.ttnn.utils_for_testing and comp_allclose
from models.common.utility_functions.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Shapes that work with the current single-block kernel (S=32, D=32 → 1 Q block,
# 1 KV block, D_t=1). Larger S or D triggers a multi-block/multi-tile CB sync
# hang (see verification_report.md). Multi-head works because each (B,H) pair is
# fully independent and processed on a single core.
PRECISION_SHAPES = [
    pytest.param((1, 1, 32, 32), id="1x1x32x32"),
    pytest.param((1, 4, 32, 32), id="1x4x32x32"),
    pytest.param((2, 4, 32, 32), id="2x4x32x32"),
]


@pytest.mark.parametrize("shape", PRECISION_SHAPES)
def test_scaled_dot_product_attention_precision_baseline(device, shape):
    """Measure PCC, max abs error, mean abs error, relative RMS error.

    Compares the TTNN Flash Attention output against the PyTorch reference
    (torch.nn.functional.scaled_dot_product_attention) in bf16.
    """
    torch.manual_seed(42)
    B, H, S, D = shape
    dtype = torch.bfloat16

    q = torch.randn(B, H, S, D, dtype=dtype)
    k = torch.randn(B, H, S, D, dtype=dtype)
    v = torch.randn(B, H, S, D, dtype=dtype)

    # PyTorch reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # TTNN
    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    output_torch = ttnn.to_torch(output)

    # PCC check
    pcc_threshold = 0.995
    assert_with_pcc(ref, output_torch, pcc=pcc_threshold)

    # Detailed metrics
    ref_f = ref.to(torch.float32)
    out_f = output_torch.to(torch.float32)
    max_abs_err = (ref_f - out_f).abs().max().item()
    mean_abs_err = (ref_f - out_f).abs().mean().item()
    rms_err = ((ref_f - out_f) ** 2).mean().sqrt().item()
    ref_std = ref_f.std().item()
    rel_rms_err = rms_err / ref_std if ref_std > 0 else float("inf")

    # comp_allclose for atol/rtol check
    allclose_pass, allclose_msg = comp_allclose(ref_f, out_f, rtol=0.1, atol=0.05)
    assert allclose_pass, f"comp_allclose failed: {allclose_msg}"

    print(
        f"\n  shape={shape}: PCC≥{pcc_threshold} ✓ "
        f"max_abs={max_abs_err:.6f} mean_abs={mean_abs_err:.6f} "
        f"rms={rms_err:.6f} rel_rms={rel_rms_err:.6f}"
    )
