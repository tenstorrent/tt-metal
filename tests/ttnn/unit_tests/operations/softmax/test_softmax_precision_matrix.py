# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision matrix for ttnn.operations.softmax.softmax — Refinement 2.

The precision axis is a **bundled** (input dtype, math_fidelity,
fp32_dest_acc_en) triple, per eval/golden_tests/softmax/feature_spec.py.
Refinement 2 adds four bf16 names on top of the Phase-0 fp32 name:

    fp32_hifi4_fp32acc   (Phase 0)
    bf16_hifi2_fp32acc   (Refinement 2)
    bf16_hifi2_bf16acc   (Refinement 2)
    bf16_hifi4_fp32acc   (Refinement 2)
    bf16_hifi4_bf16acc   (Refinement 2)

This test:
1. Drives the op with each of the five named precision modes across a
   small shape × dim × numeric_stable matrix and checks PCC against the
   tier-keyed thresholds from eval/golden_tests/softmax/helpers.TOLERANCES
   (the same bands the golden suite uses).
2. Verifies negative cases — bf16 input paired with a (math_fidelity,
   fp32_dest_acc_en) combo not in PRECISION_CONFIG must be rejected.

The test prints a one-line summary per case with PCC, max-abs, mean-abs,
and relative RMS so the verifier can capture the measured numbers.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.softmax.feature_spec import PRECISION_CONFIG
from eval.golden_tests.softmax.helpers import TOLERANCES
from models.common.utility_functions import comp_pcc
from ttnn.operations.softmax import SUPPORTED, softmax


# ttnn → torch dtype map. bfloat16 has a torch type; bfloat8_b doesn't
# (used in the helpers' fallback path). This refinement only exercises
# fp32 + bf16 in the matrix, but we keep the map open for future modes.
_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


# A small but representative shape set:
#   - one single-tile shape (32x32)
#   - one multi-tile, multi-batch shape
#   - one wide-W shape (chunk-loop intermediate width)
SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32_single_tile"),
    pytest.param((2, 4, 32, 256), id="batched_32x256"),
    pytest.param((1, 1, 64, 1024), id="wide_64x1024"),
]


# All five precision names (the four new ones + the Phase-0 baseline so
# this test also doubles as the regression for fp32_hifi4_fp32acc).
PRECISION_NAMES = [pytest.param(name, id=name) for name in SUPPORTED["precision"]]


def _build_compute_kernel_config(precision_name: str) -> ttnn.ComputeConfigDescriptor:
    spec = PRECISION_CONFIG[precision_name]
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=spec["math_fidelity"],
        fp32_dest_acc_en=spec["fp32_dest_acc_en"],
        math_approx_mode=False,
    )


def _relative_rms(expected: torch.Tensor, actual: torch.Tensor) -> float:
    diff = (expected - actual).flatten().to(torch.float64)
    ref_std = expected.flatten().to(torch.float64).std().item()
    if ref_std == 0:
        return float("nan")
    return (diff.pow(2).mean().sqrt() / ref_std).item()


@pytest.mark.parametrize("precision_name", PRECISION_NAMES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_precision_matrix(device, precision_name, shape, dim, numeric_stable, capsys):
    """Run softmax under each supported precision and verify PCC against
    the golden-suite TOLERANCES band for that precision name."""
    spec = PRECISION_CONFIG[precision_name]
    ttnn_dtype = spec["ttnn_dtype"]
    torch_dtype = _TORCH_DTYPE[ttnn_dtype]
    compute_kernel_config = _build_compute_kernel_config(precision_name)

    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch_dtype)
    # Reference is always computed in fp32 then cast back, matching the
    # golden helpers.pytorch_softmax contract.
    torch_expected = torch.softmax(torch_input.to(torch.float32), dim=dim).to(torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(
        ttnn_input,
        dim=dim,
        numeric_stable=numeric_stable,
        compute_kernel_config=compute_kernel_config,
    )

    # Output shape / dtype / layout must mirror the input contract.
    assert ttnn_output.shape == ttnn_input.shape
    assert ttnn_output.dtype == ttnn_dtype
    assert ttnn_output.layout == ttnn.TILE_LAYOUT

    torch_output = ttnn.to_torch(ttnn_output)

    # Compare in fp64 to avoid the metric itself losing precision on
    # bf16 outputs. We assert against TOLERANCES[precision_name].pcc;
    # everything else is observability-only.
    expected64 = torch_expected.to(torch.float64)
    output64 = torch_output.to(torch.float64)

    pcc_threshold, rms_threshold = TOLERANCES[precision_name]

    _, pcc_val = comp_pcc(expected64, output64, pcc=pcc_threshold)
    abs_err = (expected64 - output64).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms_rel = _relative_rms(expected64, output64)

    summary = (
        f"PRECISION_MATRIX precision={precision_name} shape={tuple(shape)} "
        f"dim={dim} numeric_stable={numeric_stable} "
        f"pcc={pcc_val:.7f} (>= {pcc_threshold}) "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rms_rel={rms_rel:.3e} "
        f"(rms_threshold={rms_threshold})"
    )
    with capsys.disabled():
        print(summary)

    assert pcc_val >= pcc_threshold, (
        f"PCC {pcc_val:.7f} below threshold {pcc_threshold} for {precision_name} "
        f"shape={shape} dim={dim} numeric_stable={numeric_stable}"
    )


# --------------------------------------------------------------------------
# Negative cases — bf16 input paired with a (fidelity, fp32_dest_acc_en) combo
# not in PRECISION_CONFIG must be rejected by validate().
# --------------------------------------------------------------------------
VALIDATION_ERRORS = (NotImplementedError, ValueError, RuntimeError)


@pytest.mark.parametrize(
    "math_fidelity, fp32_dest_acc_en, reason",
    [
        # HiFi3 + bf16 not in PRECISION_CONFIG (only HiFi2 / HiFi4 ship for bf16).
        pytest.param(ttnn.MathFidelity.HiFi3, True, "HiFi3_no_precision_name", id="bf16_HiFi3_fp32acc"),
        pytest.param(ttnn.MathFidelity.HiFi3, False, "HiFi3_no_precision_name", id="bf16_HiFi3_bf16acc"),
        # LoFi + bf16 not in PRECISION_CONFIG either.
        pytest.param(ttnn.MathFidelity.LoFi, True, "LoFi_no_precision_name", id="bf16_LoFi_fp32acc"),
        pytest.param(ttnn.MathFidelity.LoFi, False, "LoFi_no_precision_name", id="bf16_LoFi_bf16acc"),
    ],
)
def test_softmax_rejects_bf16_with_unbundled_compute_config(device, math_fidelity, fp32_dest_acc_en, reason):
    """bf16 input + (math_fidelity, fp32_dest_acc_en) combo not in PRECISION_CONFIG
    must resolve to precision=None and be rejected by validate()."""
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1, compute_kernel_config=bad_config)


# --------------------------------------------------------------------------
# bf16 accept tests — exercises the four new precision names against the
# default-None path indirectly (the only None path resolves to
# fp32_hifi4_fp32acc, so each bf16 mode must be passed explicitly).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "precision_name",
    [
        "bf16_hifi2_fp32acc",
        "bf16_hifi2_bf16acc",
        "bf16_hifi4_fp32acc",
        "bf16_hifi4_bf16acc",
    ],
)
def test_softmax_accepts_each_bf16_precision_mode(device, precision_name):
    """Smoke test: every new bf16 precision mode must accept a small bf16 input
    and return a tensor of correct shape/dtype/layout. Numerical bands are the
    job of test_softmax_precision_matrix above; this guards the API surface."""
    spec = PRECISION_CONFIG[precision_name]
    config = _build_compute_kernel_config(precision_name)
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=spec["ttnn_dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-1, compute_kernel_config=config)
    assert ttnn_output.shape == ttnn_input.shape
    assert ttnn_output.dtype == spec["ttnn_dtype"]
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
