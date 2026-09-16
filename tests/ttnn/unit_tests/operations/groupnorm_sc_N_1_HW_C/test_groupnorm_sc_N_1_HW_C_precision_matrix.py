# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for groupnorm_sc_N_1_HW_C (Refinement 3 — /numeric-formats-metal §10).

The single authoritative precision characterisation: every SUPPORTED dtype x every `math_fidelity` x
`fp32_dest_acc_en in {True, False}` x {uniform, normal} inputs over 8 shapes (tile-aligned, HW-non-aligned,
C-non-aligned, both, multi-image, group-straddling). Every metric of §11 is printed for every cell (one
`PRECISION_MATRIX ...` line per case, machine-readable for precision_matrix_results.md); only PCC is asserted.

Cells the op declares in EXCLUSIONS_COMPUTE_CONFIG (fp32 input + 16-bit DEST, the precision convention's mandatory
refusal) are skipped here and pinned as a refusal by `test_fp32_input_16bit_dest_is_refused`.

PCC thresholds (skill §11): default config (HiFi4 + fp32 DEST) shapes 0.999 for bf16 / fp32; the matrix (every
fidelity, both DEST widths) 0.99; bf8b input 0.99 everywhere (block-float input precision).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose, comp_pcc
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.groupnorm_sc_N_1_HW_C import (
    EXCLUSIONS_COMPUTE_CONFIG,
    SUPPORTED_COMPUTE_CONFIG,
    default_compute_kernel_config,
    groupnorm_sc_N_1_HW_C,
)

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def pytorch_reference(x, num_groups, *, gamma=None, beta=None, eps=1e-5):
    """GroupNorm on (N, 1, HW, C): fp32 math through torch's (N, C, HW) group_norm."""
    xf = x.to(torch.float32)
    N, _, HW, C = xf.shape
    x_nchw = xf.squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    out = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return out.permute(0, 2, 1).unsqueeze(1)


def _pcc_threshold(dtype, math_fidelity, fp32_acc):
    if dtype == ttnn.bfloat8_b:
        return 0.99
    if math_fidelity == ttnn.MathFidelity.HiFi4 and fp32_acc:
        return 0.999  # default-config precision point
    return 0.99


def _is_excluded(dtype, fp32_acc):
    cell = {"dtype": dtype, "fp32_dest_acc_en": fp32_acc}
    return any(all(cell.get(k) == v for k, v in exc.items()) for exc in EXCLUSIONS_COMPUTE_CONFIG)


SHAPES = [
    # (shape, num_groups) — (N, 1, HW, C)
    pytest.param((1, 1, 32, 32), 1, id="32x32_small"),
    pytest.param((1, 1, 32, 64), 2, id="32x64"),
    pytest.param((1, 1, 64, 128), 4, id="64x128"),
    pytest.param((1, 1, 128, 512), 8, id="128x512"),
    pytest.param((1, 1, 1024, 640), 32, id="1024x640_sd_straddle_large"),
    pytest.param((1, 1, 32, 48), 2, id="32x48_C_non_aligned"),
    pytest.param((1, 1, 48, 64), 1, id="48x64_HW_non_aligned"),
    pytest.param((2, 1, 48, 80), 4, id="2x48x80_both_non_aligned_batch2"),
]


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape,num_groups", SHAPES)
def test_groupnorm_sc_N_1_HW_C_precision_matrix(
    device, shape, num_groups, dtype, math_fidelity, fp32_acc, distribution
):
    assert fp32_acc in SUPPORTED_COMPUTE_CONFIG["fp32_dest_acc_en"]
    if _is_excluded(dtype, fp32_acc):
        pytest.skip("EXCLUSIONS_COMPUTE_CONFIG: fp32 input + 16-bit DEST (precision convention: lossy, refused)")

    base = default_compute_kernel_config()
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=base.math_approx_mode,
        dst_full_sync_en=base.dst_full_sync_en,
    )

    torch.manual_seed(0)
    C = shape[-1]
    gen = torch.rand if distribution == "rand" else torch.randn
    torch_x = gen(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    torch_gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    torch_beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_beta = ttnn.from_torch(torch_beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # bf8b quantises the input on upload: the reference must see what the device saw
    x_seen = ttnn.to_torch(tt_x).to(torch.float32) if dtype == ttnn.bfloat8_b else torch_x
    expected = pytorch_reference(x_seen, num_groups, gamma=torch_gamma, beta=torch_beta)

    out = groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_gamma, beta=tt_beta, compute_kernel_config=config)
    assert out.dtype == dtype
    got = ttnn.to_torch(out).to(torch.float32)
    assert torch.isfinite(got).all(), "output contains NaN/Inf"

    # --- §11 metrics, printed for every case ---
    exp32 = expected.to(torch.float32)
    _, pcc_val = comp_pcc(exp32, got, 0.0)  # (passed, pcc) — the threshold is applied below
    pcc_val = float(pcc_val)
    _, allclose_msg = comp_allclose(exp32, got, rtol=0.05, atol=0.05)
    ulp = calculate_detailed_ulp_stats(exp32, got)
    abs_err = (got - exp32).abs()
    median_abs_err = abs_err.median().item()
    p99_abs_err = torch.quantile(abs_err.flatten(), 0.99).item()
    relative_rms_err = (abs_err.pow(2).mean().sqrt() / exp32.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    fid = str(math_fidelity).split(".")[-1]
    print(
        f"\nPRECISION_MATRIX shape={shape} G={num_groups} dtype={dtype} fidelity={fid} fp32_acc={fp32_acc} "
        f"dist={distribution} pcc={pcc_val:.6f} max_abs={abs_err.max().item():.4g} median_abs={median_abs_err:.4g} "
        f"p99_abs={p99_abs_err:.4g} rel_rms={relative_rms_err:.4g} ulp_max={ulp['max_ulp']:.3g} "
        f"ulp_mean={ulp['mean_ulp']:.3g} ulp_median={ulp['median_ulp']:.3g} ulp_p95={ulp['p95_ulp']:.3g} "
        f"ulp_p99={ulp['p99_ulp']:.3g} | {allclose_msg}"
    )
    threshold = _pcc_threshold(dtype, math_fidelity, fp32_acc)
    assert pcc_val >= threshold, f"PCC {pcc_val:.6f} < {threshold}"


def test_fp32_input_16bit_dest_is_refused(device, expect_error):
    """Precision convention: fp32 input + fp32_dest_acc_en=False is an EXCLUSIONS cell (refused, never silently run)."""
    assert {"dtype": ttnn.float32, "fp32_dest_acc_en": False} in EXCLUSIONS_COMPUTE_CONFIG
    tt_x = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error(ExcludedCell, "fp32_dest_acc_en"):
        groupnorm_sc_N_1_HW_C(tt_x, 2, compute_kernel_config=config)


def test_caller_fidelity_is_honoured_not_gated(device):
    """math_fidelity is not a registry axis: any value is legal and the op must run it (LoFi + 16-bit DEST here)."""
    torch.manual_seed(0)
    torch_x = torch.randn(1, 1, 64, 64).to(torch.bfloat16)
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False)
    got = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, 2, compute_kernel_config=config)).to(torch.float32)
    expected = pytorch_reference(torch_x, 2)
    passed, pcc_val = comp_pcc(expected, got, 0.99)
    assert passed, f"PCC {pcc_val}"
