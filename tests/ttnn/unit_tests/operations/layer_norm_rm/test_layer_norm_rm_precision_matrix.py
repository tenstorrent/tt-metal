# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision matrix for ttnn.operations.layer_norm_rm.layer_norm — Refinement 1.

The precision axis is a **bundled** (input dtype, math_fidelity,
fp32_dest_acc_en) triple, per eval/golden_tests/layer_norm_rm/feature_spec.py.
Refinement 1 adds three precision names on top of the Phase-0 fp32 name:

    fp32_hifi4_fp32acc   (Phase 0)
    bf16_hifi4_fp32acc   (Refinement 1)
    bf16_hifi4_bf16acc   (Refinement 1)
    bf8b_hifi4_bf16acc   (Refinement 1 — listed in SUPPORTED but unreachable
                          through layer_norm_rm until Refinement 2 lifts the
                          ROW_MAJOR-only restriction on input layout)

This test:
1. Drives the op with each of the three reachable precision modes (fp32 and
   the two bf16 modes) across a small shape × affine matrix and checks PCC
   against the tier-keyed thresholds from
   eval/golden_tests/layer_norm_rm/helpers.TOLERANCES (the same bands the
   golden suite uses).
2. Exercises mixed-dtype combinations (bf16 input + fp32 gamma, fp32 input +
   bf16 gamma) — these go through the dtype-aware reader CT-arg split
   (input_chunk_bytes vs affine_chunk_bytes).
3. Verifies negative cases: bf16 input paired with a (math_fidelity,
   fp32_dest_acc_en) combo not in PRECISION_CONFIG must be rejected.

The test prints a one-line summary per case with PCC, max-abs, mean-abs,
and relative RMS so the verifier can capture the measured numbers.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.layer_norm_rm.feature_spec import PRECISION_CONFIG
from eval.golden_tests.layer_norm_rm.helpers import TOLERANCES
from models.common.utility_functions import comp_pcc
from ttnn.operations.layer_norm_rm import SUPPORTED, layer_norm


# ttnn → torch dtype map. bf8b doesn't have a torch counterpart (and the
# layer_norm_rm op rejects bf8b input today because SUPPORTED["layout"] is
# ROW_MAJOR-only and bf8b in RM is INVALID); the precision matrix only
# exercises fp32 + bf16 input dtypes.
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


# Only precision names whose input dtype is actually reachable for
# layer_norm_rm right now (RM-only layout). bf8b_hifi4_bf16acc is in
# SUPPORTED but its input is INVALID under RM layout — skipping it here
# keeps the test honest about what runs.
REACHABLE_PRECISION_NAMES = [
    name for name in SUPPORTED["precision"] if PRECISION_CONFIG[name]["ttnn_dtype"] in _TORCH_DTYPE
]
PRECISION_PARAMS = [pytest.param(name, id=name) for name in REACHABLE_PRECISION_NAMES]


# Affine modes — gamma/beta presence sweep.
AFFINE_MODES = [
    pytest.param("no_affine", id="affine=none"),
    pytest.param("gamma_only", id="affine=gamma_only"),
    pytest.param("gamma_beta", id="affine=gamma_beta"),
]


# Affine dtypes — bf8b excluded for the same RM-layout reason as the input.
AFFINE_DTYPES = [
    pytest.param(ttnn.float32, id="affine_dtype=fp32"),
    pytest.param(ttnn.bfloat16, id="affine_dtype=bf16"),
]


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


def _pytorch_layer_norm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5):
    """fp32-internal reference; mirrors helpers.pytorch_layer_norm."""
    original_dtype = input_tensor.dtype
    x = input_tensor.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    if beta is not None:
        y = y + beta.to(torch.float32).reshape(-1)
    return y.to(original_dtype)


@pytest.mark.parametrize("precision_name", PRECISION_PARAMS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
@pytest.mark.parametrize("affine_dtype", AFFINE_DTYPES)
def test_layer_norm_rm_precision_matrix(device, precision_name, shape, affine_mode, affine_dtype, capsys):
    """Run layer_norm under each supported precision (+ affine_dtype) and verify
    PCC against the golden-suite TOLERANCES band for that precision name."""
    spec = PRECISION_CONFIG[precision_name]
    ttnn_dtype = spec["ttnn_dtype"]
    torch_dtype = _TORCH_DTYPE[ttnn_dtype]
    torch_affine_dtype = _TORCH_DTYPE[affine_dtype]
    compute_kernel_config = _build_compute_kernel_config(precision_name)

    W = shape[-1]
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch_dtype)

    torch_gamma = None
    torch_beta = None
    ttnn_gamma = None
    ttnn_beta = None

    if affine_mode in ("gamma_only", "gamma_beta"):
        torch_gamma = torch.randn(W, dtype=torch_affine_dtype) * 0.5 + 1.0
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=affine_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if affine_mode == "gamma_beta":
        torch_beta = torch.randn(W, dtype=torch_affine_dtype) * 0.1
        ttnn_beta = ttnn.from_torch(
            torch_beta.reshape(1, 1, 1, W),
            dtype=affine_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    torch_expected = _pytorch_layer_norm(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(
        ttnn_input,
        ttnn_gamma,
        ttnn_beta,
        epsilon=1e-5,
        compute_kernel_config=compute_kernel_config,
    )

    # Output shape / dtype / layout mirror the input contract.
    assert ttnn_output.shape == ttnn_input.shape
    assert ttnn_output.dtype == ttnn_dtype
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT

    torch_output = ttnn.to_torch(ttnn_output)

    # Compare in fp64 to avoid the metric itself losing precision on bf16
    # outputs. Assert against TOLERANCES[precision_name].pcc; everything
    # else is observability-only.
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
        f"affine={affine_mode} affine_dtype={affine_dtype} "
        f"pcc={pcc_val:.7f} (>= {pcc_threshold}) "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rms_rel={rms_rel:.3e} "
        f"(rms_threshold={rms_threshold})"
    )
    with capsys.disabled():
        print(summary)

    assert pcc_val >= pcc_threshold, (
        f"PCC {pcc_val:.7f} below threshold {pcc_threshold} for {precision_name} "
        f"shape={shape} affine={affine_mode} affine_dtype={affine_dtype}"
    )


# --------------------------------------------------------------------------
# Negative cases — bf16 input paired with a (fidelity, fp32_dest_acc_en) combo
# not in PRECISION_CONFIG must be rejected by validate().
# --------------------------------------------------------------------------
VALIDATION_ERRORS = (NotImplementedError, ValueError, RuntimeError)


@pytest.mark.parametrize(
    "math_fidelity, fp32_dest_acc_en",
    [
        # HiFi3 + bf16 not in PRECISION_CONFIG (only HiFi4 ships for bf16).
        pytest.param(ttnn.MathFidelity.HiFi3, True, id="bf16_HiFi3_fp32acc"),
        pytest.param(ttnn.MathFidelity.HiFi3, False, id="bf16_HiFi3_bf16acc"),
        # HiFi2 + bf16 not in PRECISION_CONFIG either.
        pytest.param(ttnn.MathFidelity.HiFi2, True, id="bf16_HiFi2_fp32acc"),
        pytest.param(ttnn.MathFidelity.HiFi2, False, id="bf16_HiFi2_bf16acc"),
        # LoFi + bf16 not in PRECISION_CONFIG.
        pytest.param(ttnn.MathFidelity.LoFi, True, id="bf16_LoFi_fp32acc"),
        pytest.param(ttnn.MathFidelity.LoFi, False, id="bf16_LoFi_bf16acc"),
    ],
)
def test_layer_norm_rm_rejects_bf16_with_unbundled_compute_config(device, math_fidelity, fp32_dest_acc_en):
    """bf16 input + (math_fidelity, fp32_dest_acc_en) combo not in
    PRECISION_CONFIG must resolve to precision=None and be rejected by
    validate()."""
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
    )
    with pytest.raises(VALIDATION_ERRORS):
        layer_norm(ttnn_input, compute_kernel_config=bad_config)


# --------------------------------------------------------------------------
# bf16 accept tests — each of the new bf16 precision modes must accept a
# small bf16 input and return a tensor of correct shape/dtype/layout.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "precision_name",
    [
        "bf16_hifi4_fp32acc",
        "bf16_hifi4_bf16acc",
    ],
)
def test_layer_norm_rm_accepts_each_bf16_precision_mode(device, precision_name):
    """Smoke test: every new bf16 precision mode accepts a small bf16 input."""
    spec = PRECISION_CONFIG[precision_name]
    config = _build_compute_kernel_config(precision_name)
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=spec["ttnn_dtype"],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = layer_norm(ttnn_input, compute_kernel_config=config)
    assert ttnn_output.shape == ttnn_input.shape
    assert ttnn_output.dtype == spec["ttnn_dtype"]
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT


# --------------------------------------------------------------------------
# Mixed-dtype smoke test — input and affine tensors with different dtypes.
# This exercises the dtype-aware reader CT-arg split
# (input_chunk_bytes vs affine_chunk_bytes).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_precision, affine_dtype",
    [
        pytest.param("fp32_hifi4_fp32acc", ttnn.bfloat16, id="fp32_input_bf16_affine"),
        pytest.param("bf16_hifi4_fp32acc", ttnn.float32, id="bf16_input_fp32_affine"),
        pytest.param("bf16_hifi4_bf16acc", ttnn.float32, id="bf16_bf16acc_fp32_affine"),
    ],
)
def test_layer_norm_rm_mixed_dtype_input_and_affine(device, input_precision, affine_dtype):
    """When input dtype ≠ affine dtype, the reader must use different byte
    strides for the input vs gamma/beta reads. This is the case that gets
    silently corrupted if input_chunk_bytes and affine_chunk_bytes aren't
    split."""
    spec = PRECISION_CONFIG[input_precision]
    ttnn_dtype = spec["ttnn_dtype"]
    torch_dtype = _TORCH_DTYPE[ttnn_dtype]
    torch_affine_dtype = _TORCH_DTYPE[affine_dtype]
    config = _build_compute_kernel_config(input_precision)

    shape = (1, 1, 32, 128)
    W = shape[-1]
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch_dtype)
    torch_gamma = torch.randn(W, dtype=torch_affine_dtype) * 0.5 + 1.0
    torch_beta = torch.randn(W, dtype=torch_affine_dtype) * 0.1

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, W),
        dtype=affine_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta.reshape(1, 1, 1, W),
        dtype=affine_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_expected = _pytorch_layer_norm(
        torch_input,
        torch_gamma,
        torch_beta,
        epsilon=1e-5,
    )

    ttnn_output = layer_norm(
        ttnn_input,
        ttnn_gamma,
        ttnn_beta,
        epsilon=1e-5,
        compute_kernel_config=config,
    )

    torch_output = ttnn.to_torch(ttnn_output)
    pcc_threshold, _ = TOLERANCES[input_precision]
    _, pcc_val = comp_pcc(
        torch_expected.to(torch.float64),
        torch_output.to(torch.float64),
        pcc=pcc_threshold,
    )
    assert pcc_val >= pcc_threshold, (
        f"Mixed-dtype PCC {pcc_val:.7f} below threshold {pcc_threshold} for "
        f"input_precision={input_precision} affine_dtype={affine_dtype}"
    )
