# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# test_rms_norm_precision_matrix — the single authoritative precision
# characterization test for rms_norm (Refinement 2: numeric-formats skill §10).
#
# Cross-product over:
#   * shapes (tile-aligned only — ROW_MAJOR / non-aligned are Refinement 3),
#     covering both Regime A (row-parallel) and Regime B (wide-W cross-core),
#   * dtype       in SUPPORTED["dtype"]            = {bf16, fp32, bf8b}
#   * math_fidelity in {HiFi4, HiFi3, HiFi2, LoFi} (NOT gated by the op)
#   * fp32_dest_acc_en in {True, False}
#   * input distribution in {uniform, normal}
#
# The EXCLUSIONS cell {dtype=float32, fp32_dest_acc_en=False} is skipped with
# the EXCLUSIONS reason (fp32 input requires fp32 accumulation).
#
# Asserts on PCC only; prints every metric for observability. PCC thresholds
# follow the numeric-formats skill §11.

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations._op_contract import SupportRefusal
from models.common.utility_functions import comp_pcc, comp_allclose


_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}

# PCC floor per dtype (skill §11; precision-matrix band is looser than the
# default-config baseline since it sweeps LoFi + bf16 accumulation).
_PCC_FLOOR = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.99,
    ttnn.bfloat8_b: 0.98,  # block-float across LoFi is inherently lower
}


def _reference(x, gamma, eps):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


@pytest.mark.parametrize(
    "distribution",
    [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")],
)
@pytest.mark.parametrize(
    "fp32_acc",
    [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")],
)
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
@pytest.mark.parametrize(
    "gamma_mode",
    [pytest.param("gamma", id="gamma"), pytest.param("no_gamma", id="no_gamma")],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32_small"),
        pytest.param((32, 64), id="32x64"),
        pytest.param((64, 128), id="64x128"),
        pytest.param((128, 512), id="128x512_regimeB"),
        pytest.param((4, 8, 32, 256), id="4x8x32x256_rank4"),
        pytest.param((1024, 1024), id="1024x1024_large"),
        pytest.param((256, 2048), id="256x2048_large"),
        pytest.param((1, 1, 32, 4096), id="1x1x32x4096_wide_regimeB"),
    ],
)
def test_rms_norm_precision_matrix(device, shape, gamma_mode, dtype, math_fidelity, fp32_acc, distribution):
    # EXCLUSIONS: fp32 input mandates fp32 accumulation.
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: {float32, fp32_dest_acc_en=False} — fp32 needs fp32 accumulation")

    eps = 1e-6
    torch.manual_seed(0)
    torch_dtype = _TORCH_DTYPE[dtype]
    gen = torch.rand if distribution == "rand" else torch.randn
    torch_input = gen(*shape).to(torch_dtype)

    torch_gamma = None
    ttnn_gamma = None
    if gamma_mode == "gamma":
        W = shape[-1]
        torch_gamma = gen(W).to(torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    expected = _reference(torch_input, torch_gamma, eps)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=eps, compute_kernel_config=cfg)
    got = ttnn.to_torch(ttnn_out).float()

    # --- metrics (printed for every case, asserted on PCC only) ---
    pcc_pass, pcc_msg = comp_pcc(expected, got, _PCC_FLOOR[dtype])
    _, allclose_msg = comp_allclose(expected, got)
    abs_err = (got - expected).abs()
    median_abs = abs_err.median().item()
    p99_abs = torch.quantile(abs_err.flatten(), 0.99).item()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    n_inf = torch.isinf(got).sum().item()
    n_nan = torch.isnan(got).sum().item()

    print(
        f"[{shape} {gamma_mode} {dtype} {math_fidelity} fp32acc={fp32_acc} {distribution}] "
        f"{pcc_msg} | {allclose_msg} | median_abs={median_abs:.5g} p99_abs={p99_abs:.5g} "
        f"relRMS={rel_rms:.5g} inf={n_inf} nan={n_nan}"
    )

    assert n_inf == 0 and n_nan == 0, f"non-finite output: inf={n_inf} nan={n_nan}"
    assert pcc_pass, f"PCC below {_PCC_FLOOR[dtype]}: {pcc_msg}"


def test_rms_norm_precision_matrix_fp32_no_acc_refused(device):
    """The documented EXCLUSION cell raises a support refusal, not garbage."""
    x = ttnn.from_torch(torch.randn(32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    with pytest.raises(SupportRefusal):
        rms_norm(x, compute_kernel_config=cfg)
