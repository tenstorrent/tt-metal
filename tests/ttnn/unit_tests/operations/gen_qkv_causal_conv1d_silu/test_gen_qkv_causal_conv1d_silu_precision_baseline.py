# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for gen_qkv_causal_conv1d_silu (verifier artifact).

Measures, per golden cell (BB-22: only golden cells), over q, k and v concatenated along channels:
PCC (comp_pcc), max abs error (comp_allclose), mean abs error, relative RMS error (RMS error over
the reference's std, as eval.metrics.check_output), bf16 ULP distance against the float64 reference
rounded to bf16 (p99 over all elements; max over the elements away from zero), and the got/true
ratio spread r = actual / expected. "Away from zero" = |expected| >= 10% of the reference's std, for
both the ULP max and the ratio (the scale-bug detector: a tight cluster around a non-1.0 constant
means a scale/structural bug, a broad spread around 1.0 is rounding noise).

Gated on PCC and relative RMS together from the golden helpers' TOLERANCES (HiFi4 cells only), plus
the per-element FP32_DEST_BOUND on the fp32-DEST cell. Run with -s to see the table rows:

    scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/gen_qkv_causal_conv1d_silu/\
test_gen_qkv_causal_conv1d_silu_precision_baseline.py -s
"""

from __future__ import annotations

import re

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from eval.golden_tests.gen_qkv_causal_conv1d_silu.helpers import FP32_DEST_BOUND, TOLERANCES
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _run,
    pytorch_reference,
)

# (PHASE0_CASES index, label): small / medium / large / largest at the default config, plus the
# fp32-DEST cancelling-taps cell.
BASELINE_CASES = [
    pytest.param(0, id="conv1d-001_T32_C2560"),
    pytest.param(2, id="conv1d-003_T128_C2560"),
    pytest.param(14, id="conv1d-020_T2048_C2048"),
    pytest.param(3, id="conv1d-004_T4096_C2560"),
    pytest.param(9, id="conv1d-011_T128_fp32dest_cancelling"),
]


def _bf16_ulp(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Per-element ULP distance at bfloat16 granularity (monotonic integer ordering of the bits)."""

    def ordered(x):
        bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        return torch.where(bits < 0, -(bits + (1 << 15)), bits)

    return (ordered(actual) - ordered(expected)).abs().float()


@pytest.mark.parametrize("case_index", BASELINE_CASES)
def test_precision_baseline(device, case_index):
    case = PHASE0_CASES[case_index].values[0]
    outputs, host, _ = _run(device, case)
    (_, seq, _), widths, _, _, _ = case["inputs"]
    _, conv_input, history, taps = host
    refs, sums = pytorch_reference(conv_input, history, taps, widths)

    actual = torch.cat([ttnn.to_torch(o).reshape(1, seq, -1).double() for o in outputs], dim=-1)
    expected = torch.cat(refs, dim=-1)
    terms = torch.cat(sums, dim=-1)

    _, pcc = comp_pcc(expected, actual)
    _, allclose_msg = comp_allclose(expected, actual)
    max_abs = float(re.search(r"Max ATOL Delta: ([0-9.eE+-]+|nan|inf)", allclose_msg).group(1))
    diff = actual - expected
    mean_abs = float(diff.abs().mean())
    rel_rms = float(diff.pow(2).mean().sqrt() / expected.std())
    ulp = _bf16_ulp(actual, expected)
    ulp_p99 = float(ulp.flatten().kthvalue(max(1, int(0.99 * ulp.numel()))).values)

    # Away from zero only: a near-zero output whose sign differs from the reference's spans the
    # whole bf16 ordering (tens of thousands of "ULP") without being a large error.
    mask = expected.abs() >= 0.1 * expected.std()
    ulp_max = float(ulp[mask].max())
    ratio = (actual[mask] / expected[mask]).float()
    r_med = float(ratio.median())
    r_p5 = float(ratio.kthvalue(max(1, int(0.05 * ratio.numel()))).values)
    r_p95 = float(ratio.kthvalue(max(1, int(0.95 * ratio.numel()))).values)

    fp32 = case["fp32"] and not case["config_none"]
    print(
        f"\nPRECISION_BASELINE {PHASE0_CASES[case_index].id}: fp32_dest={fp32} pcc={pcc:.6f} "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.5f} rel_rms={rel_rms:.5f} "
        f"ulp_p99={ulp_p99:.0f} ulp_max_off_zero={ulp_max:.0f} ratio_median={r_med:.5f} "
        f"ratio_p5={r_p5:.5f} ratio_p95={r_p95:.5f}"
    )

    pcc_min, rms_max = TOLERANCES[(ttnn.bfloat16, bool(fp32))]
    assert pcc >= pcc_min, f"PCC {pcc} < {pcc_min}"
    assert rel_rms <= rms_max, f"relative RMS {rel_rms} > {rms_max}"
    # Scale-bug detector: the bulk of the ratio distribution must sit on 1.0.
    assert abs(r_med - 1.0) < 0.01, f"got/true ratio median {r_med} is off 1.0 (scale/structural bug?)"
    if fp32:
        relative, absolute = FP32_DEST_BOUND
        bound_ratio = diff.abs() / (relative * expected.abs() + absolute * terms)
        assert float(bound_ratio.max()) <= 1.0, f"fp32 bound exceeded: {float(bound_ratio.max())}"
