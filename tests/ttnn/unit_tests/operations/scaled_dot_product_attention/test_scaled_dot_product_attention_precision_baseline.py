# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Flash Attention).

Measures PCC, max abs error, mean abs error, and relative RMS error across
3-4 representative shapes. Uses assert_with_pcc and comp_allclose from the
shared test utilities — no hand-computed metrics.

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_precision_baseline.py
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# Representative shapes: small, medium, large, multi-head
SHAPES = [
    # (label, Q_shape, K_shape, V_shape)
    ("small_32x32", (1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    ("medium_128x64", (1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    ("large_256x64", (1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    ("multi_head_8x128x64", (1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)),
]


@pytest.mark.parametrize(
    "label, q_shape, k_shape, v_shape",
    SHAPES,
    ids=[s[0] for s in SHAPES],
)
def test_sdpa_precision_baseline(device, label, q_shape, k_shape, v_shape):
    """Measure PCC, abs error, and RMS error for SDPA against torch SDPA."""
    torch.manual_seed(42)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    D = q_shape[-1]
    scale = 1.0 / math.sqrt(D)

    # PyTorch reference (fp32 internally)
    expected = torch.nn.functional.scaled_dot_product_attention(
        torch_Q.float(), torch_K.float(), torch_V.float(), scale=scale
    )

    ttnn_Q = ttnn.from_torch(
        torch_Q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_K = ttnn.from_torch(
        torch_K,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_V = ttnn.from_torch(
        torch_V,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, scale=scale)
    torch_output = ttnn.to_torch(ttnn_output)

    output_f32 = torch_output.float()
    expected_f32 = expected.float()

    # PCC
    pcc_passed, pcc_val = comp_pcc(expected_f32, output_f32, pcc=0.995)

    # abs errors
    abs_err = (output_f32 - expected_f32).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    # Relative RMS error (normalized by reference output's stddev)
    ref_std = expected_f32.std().item()
    rms_err = (output_f32 - expected_f32).pow(2).mean().sqrt().item()
    rel_rms_err = rms_err / ref_std if ref_std > 0 else float("inf")

    # allclose
    allclose_passed, allclose_msg = comp_allclose(expected_f32, output_f32, rtol=0.05, atol=0.01)

    print(f"\n[{label}] shape={q_shape}")
    print(f"  PCC: {pcc_val:.6f} (pass={pcc_passed})")
    print(f"  Max abs err: {max_abs_err:.6f}")
    print(f"  Mean abs err: {mean_abs_err:.6f}")
    print(f"  Relative RMS err: {rel_rms_err:.6f}")
    print(f"  Allclose (rtol=0.05, atol=0.01): {allclose_passed}")

    # Assert with PCC threshold
    assert_with_pcc(expected_f32, output_f32, pcc=0.995)
