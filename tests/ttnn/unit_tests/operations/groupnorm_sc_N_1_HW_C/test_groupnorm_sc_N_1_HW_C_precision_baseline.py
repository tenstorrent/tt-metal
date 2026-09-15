# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for groupnorm_sc_N_1_HW_C (verifier artefact).

Measures PCC, max / mean absolute error, relative RMS error, bf16-ULP error and the got/true
ratio spread (the scale-bug detector: a tight cluster of ``actual / expected`` around a non-1.0
constant is a uniform scale / structural bug, a broad spread centred on 1.0 is rounding noise)
over four shapes spanning the supported regimes: single tile, SD group-straddling, multi-batch
and the largest SDXL shape. The last case is offset-heavy (|mean| >> std) and probes the
``E[x^2] - mean^2`` cancellation the design's deferred ``shifted_two_pass_variance`` row covers.

The numbers are recorded in verification_report.md / changelog.md; the asserts are loose
regression guards, not tolerances.
"""

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def pytorch_reference(x, num_groups, *, gamma=None, beta=None, eps=1e-5):
    xf = x.to(torch.float32)
    N, one, HW, C = xf.shape
    x_nchw = xf.squeeze(1).permute(0, 2, 1)
    weight = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    bias = beta.to(torch.float32).reshape(C) if beta is not None else None
    out = torch.nn.functional.group_norm(x_nchw, num_groups, weight=weight, bias=bias, eps=eps)
    return out.permute(0, 2, 1).unsqueeze(1)


def _metrics(actual, expected):
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    diff = (a - e).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / torch.sqrt((e**2).mean()).clamp_min(1e-30))
    # bf16 ULP of the expected value: 2^(exponent - 7)
    exp2 = torch.floor(torch.log2(e.abs().clamp_min(1e-30)))
    ulp = torch.pow(2.0, exp2 - 7)
    ulp_err = diff / ulp
    # got/true ratio spread over finite, non-tiny references
    mask = e.abs() > 1e-2
    r = (a[mask] / e[mask]) if mask.any() else torch.ones(1, dtype=torch.float64)
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rms": rel_rms,
        "ulp_max": float(ulp_err.max()),
        "ulp_mean": float(ulp_err.mean()),
        "ratio_p5": float(q[0]),
        "ratio_median": float(q[1]),
        "ratio_p95": float(q[2]),
    }


CASES = [
    # (shape, num_groups, affine, input_offset, id)
    pytest.param((1, 1, 32, 32), 1, False, 0.0, id="single_tile"),
    pytest.param((1, 1, 1024, 640), 32, True, 0.0, id="sd15_straddle_cg20_affine"),
    pytest.param((4, 1, 128, 256), 8, True, 0.0, id="batch4_g8_affine"),
    pytest.param((1, 1, 16384, 320), 32, False, 0.0, id="sdxl_hw16384_c320"),
    pytest.param((1, 1, 1024, 640), 32, True, 8.0, id="sd15_offset_mean8_affine"),
]


@pytest.mark.parametrize("shape,num_groups,affine,offset", CASES)
def test_precision_baseline(device, shape, num_groups, affine, offset):
    torch.manual_seed(1234)
    C = shape[-1]
    x = (torch.randn(shape, dtype=torch.float32) + offset).to(torch.bfloat16)
    gamma = beta = None
    tg = tb = None
    if affine:
        gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        tg = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tb = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected = pytorch_reference(x, num_groups, gamma=gamma, beta=beta)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = groupnorm_sc_N_1_HW_C(tx, num_groups, gamma=tg, beta=tb)
    actual = ttnn.to_torch(out).to(torch.float32)

    assert torch.isfinite(actual).all()
    m = _metrics(actual, expected)
    _, allclose_msg = comp_allclose(expected, actual, rtol=0.05, atol=0.05)
    pcc_msg = assert_with_pcc(expected, actual, pcc=0.99)
    print(
        f"\nPRECISION shape={shape} G={num_groups} affine={affine} offset={offset}: {pcc_msg} | "
        f"max_abs={m['max_abs']:.4g} mean_abs={m['mean_abs']:.4g} rel_rms={m['rel_rms']:.4g} | "
        f"ulp_max={m['ulp_max']:.3g} ulp_mean={m['ulp_mean']:.3g} | "
        f"ratio p5/med/p95={m['ratio_p5']:.4f}/{m['ratio_median']:.4f}/{m['ratio_p95']:.4f} | {allclose_msg}"
    )
    # Regression guards (bf16 in/out, fp32 statistics): loose by design. The offset-heavy case is
    # informational: with |mean| >> std the apply's x*a_T + b_T cancels two O(|mean|*rstd*gamma) terms
    # evaluated at the FPU's tf32-class precision (measured rel_rms ~0.028, ratio spread broad and centred
    # on 1.0 = rounding, not scale) — see verification_report.md -> Recommendations for the lever.
    rel_rms_bound = 0.06 if offset else 0.02
    assert m["rel_rms"] < rel_rms_bound, m
    assert abs(m["ratio_median"] - 1.0) < 0.01, f"uniform scale offset: {m}"
