# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for scaled_dot_product_attention.

Single authoritative precision characterization test per /numeric-formats-metal §10.
Full cross-product of:
  - shapes (tile-aligned, small to large)
  - dtypes (bfloat16, float32, bfloat8_b)
  - fp32_dest_acc_en (True, False)
  - math_fidelity (HiFi4, HiFi3, HiFi2, LoFi)
  - input distribution (uniform, normal)

EXCLUSIONS mirror the op file: {dtype: float32, fp32_dest_acc_en: False} is skipped
(maxed input + non-maxed acc is rejected by the op — mirrors softmax convention).

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_precision_matrix.py
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Shapes — at least 8, ranging small to large. All tile-aligned (Phase 0 +
# Refinement 1 scope; non-aligned is Refinement 2).
# ---------------------------------------------------------------------------
SHAPES = [
    pytest.param((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32), id="32x32_small"),
    pytest.param((1, 1, 32, 64), (1, 1, 32, 64), (1, 1, 32, 64), id="32x64"),
    pytest.param((1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128), id="64x128"),
    pytest.param((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), id="128x64"),
    pytest.param((1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64), id="256x64_large"),
    pytest.param((1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64), id="multi_head_128x64"),
    pytest.param((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64), id="multi_batch"),
    pytest.param((1, 1, 512, 64), (1, 1, 512, 64), (1, 1, 512, 64), id="long_context"),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, id="bf8b"),
]

FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
    pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
    pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
]

DISTRIBUTIONS = [
    pytest.param("rand", id="uniform"),
    pytest.param("randn", id="normal"),
]

# PCC thresholds per /numeric-formats-metal §11.
# Precision matrix (all fidelities + fp32 acc) → 0.99.
# bf8b → 0.99 (block-float precision is inherently lower).
PCC_THRESHOLD = 0.99


def _make_ttnn_tensor(torch_tensor, device, dtype):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize("math_fidelity", FIDELITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q_shape,k_shape,v_shape", SHAPES)
def test_sdpa_precision_matrix(device, q_shape, k_shape, v_shape, dtype, math_fidelity, fp32_acc, distribution):
    """Precision matrix: full cross-product of dtype × fidelity × acc × distribution."""
    # Skip EXCLUSIONS cell: {dtype: float32, fp32_dest_acc_en: False}
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSION: {dtype: float32, fp32_dest_acc_en: False} — maxed input + non-maxed acc")

    torch.manual_seed(42)

    # torch dtype: bf8b has no native torch type, use bf16 for input generation
    torch_dtype = (
        torch.bfloat16 if dtype == ttnn.bfloat8_b else (torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    )

    if distribution == "rand":
        torch_Q = torch.rand(q_shape, dtype=torch_dtype)
        torch_K = torch.rand(k_shape, dtype=torch_dtype)
        torch_V = torch.rand(v_shape, dtype=torch_dtype)
    else:
        torch_Q = torch.randn(q_shape, dtype=torch_dtype)
        torch_K = torch.randn(k_shape, dtype=torch_dtype)
        torch_V = torch.randn(v_shape, dtype=torch_dtype)

    D = q_shape[-1]
    scale = 1.0 / math.sqrt(D)

    # PyTorch reference (fp32 internally)
    expected = torch.nn.functional.scaled_dot_product_attention(
        torch_Q.float(), torch_K.float(), torch_V.float(), scale=scale
    )

    ttnn_Q = _make_ttnn_tensor(torch_Q, device, dtype)
    ttnn_K = _make_ttnn_tensor(torch_K, device, dtype)
    ttnn_V = _make_ttnn_tensor(torch_V, device, dtype)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )

    ttnn_output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, scale=scale, compute_kernel_config=config)
    torch_output = ttnn.to_torch(ttnn_output)

    output_f32 = torch_output.float()
    expected_f32 = expected.float()

    # PCC
    pcc_passed, pcc_val = comp_pcc(expected_f32, output_f32, pcc=PCC_THRESHOLD)

    # abs errors
    abs_err = (output_f32 - expected_f32).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    # Relative RMS error (normalized by reference output's stddev)
    ref_std = expected_f32.std().item()
    rms_err = (output_f32 - expected_f32).pow(2).mean().sqrt().item()
    rel_rms_err = rms_err / ref_std if ref_std > 0 else float("inf")

    # allclose
    allclose_passed, allclose_msg = comp_allclose(expected_f32, output_f32, rtol=0.12, atol=0.02)

    print(f"\n[shape={q_shape} dtype={dtype} fidelity={math_fidelity} fp32_acc={fp32_acc} dist={distribution}]")
    print(f"  PCC: {pcc_val:.6f} (pass={pcc_passed})")
    print(f"  Max abs err: {max_abs_err:.6f}")
    print(f"  Mean abs err: {mean_abs_err:.6f}")
    print(f"  Relative RMS err: {rel_rms_err:.6f}")
    print(f"  Allclose (rtol=0.12, atol=0.02): {allclose_passed}")

    assert_with_pcc(expected_f32, output_f32, pcc=PCC_THRESHOLD)
