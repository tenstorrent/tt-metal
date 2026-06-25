# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix test for softmax (Refinement 1 — Numerical configurability).

Exercises the full cross-product of:
  - dtype ∈ {float32, bfloat16, bfloat8_b}
  - fp32_dest_acc_en ∈ {True, False}
  - math_fidelity ∈ {HiFi4, HiFi3, HiFi2, LoFi}
  - distribution ∈ {uniform, normal}
  - shapes (small → medium, tile-aligned)

The op is fp32-dest-only: fp32_dest_acc_en=False is rejected via EXCLUSIONS.
Those cells are skipped, not failed.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.common.utility_functions import comp_allclose


def pytorch_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input_tensor.float(), dim=dim)


SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32"),
    pytest.param((1, 1, 32, 64), id="32x64"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((2, 4, 32, 256), id="2x4x32x256"),
    pytest.param((4, 8, 64, 64), id="4x8x64x64"),
    pytest.param((1, 1, 128, 512), id="128x512"),
    pytest.param((2, 4, 128, 256), id="2x4x128x256"),
    pytest.param((1, 1, 64, 64), id="64x64"),
]

DTYPES = [
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.bfloat8_b, id="bf8b"),
]

MATH_FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
    pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
    pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
]

FP32_ACC = [
    pytest.param(True, id="fp32_acc"),
    # False is rejected by EXCLUSIONS — skipped, not tested
]

DISTRIBUTIONS = [
    pytest.param("rand", id="uniform"),
    pytest.param("randn", id="normal"),
]

# PCC threshold per dtype — precision matrix uses 0.99 for all (§11 of
# numeric-formats-metal skill: "Precision matrix (all fidelities + fp32
# acc)" threshold is 0.99, since LoFi is inherently lower precision).
PCC_THRESHOLD = {
    ttnn.float32: 0.99,
    ttnn.bfloat16: 0.99,
    ttnn.bfloat8_b: 0.99,
}


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("fp32_acc", FP32_ACC)
@pytest.mark.parametrize("math_fidelity", MATH_FIDELITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_precision_matrix(device, shape, dtype, math_fidelity, fp32_acc, distribution):
    """Full precision matrix: dtype × math_fidelity × fp32_acc × distribution."""
    torch.manual_seed(42)

    # Generate input
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    if distribution == "rand":
        torch_input = torch.rand(shape, dtype=torch_dtype)
    else:
        torch_input = torch.randn(shape, dtype=torch_dtype)

    expected = pytorch_softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )

    ttnn_output = ttnn.softmax(ttnn_input, dim=-1, compute_kernel_config=config)
    torch_output = ttnn.to_torch(ttnn_output)

    # Assert PCC
    threshold = PCC_THRESHOLD[dtype]
    assert_with_pcc(expected, torch_output, pcc=threshold)

    # Print all metrics for observability
    passes, msg = comp_allclose(expected, torch_output, rtol=1e-3, atol=1e-5)
    abs_err = (expected.float() - torch_output.float()).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    median_abs_err = abs_err.median().item()
    p99_abs_err = torch.quantile(abs_err.flatten().float(), 0.99).item()
    rms_diff = torch.sqrt((abs_err**2).mean()).item()
    rms_expected = torch.sqrt((expected.float() ** 2).mean()).item()
    rel_rms_err = rms_diff / rms_expected if rms_expected > 0 else float("inf")

    print(f"\n  shape={shape}, dtype={dtype}, fidelity={math_fidelity}, " f"fp32_acc={fp32_acc}, dist={distribution}")
    print(f"    PCC >= {threshold} (asserted)")
    print(f"    max_abs_err   = {max_abs_err:.6f}")
    print(f"    mean_abs_err  = {mean_abs_err:.6f}")
    print(f"    median_abs_err= {median_abs_err:.6f}")
    print(f"    p99_abs_err   = {p99_abs_err:.6f}")
    print(f"    rel_rms_err   = {rel_rms_err:.6f}")
    print(f"    allclose      = {passes} ({msg})")

    # Sanity: no NaN or Inf
    assert not torch_output.isnan().any(), "Output contains NaN"
    assert not torch_output.isinf().any(), "Output contains Inf"
