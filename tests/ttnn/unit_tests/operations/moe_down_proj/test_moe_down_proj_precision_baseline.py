# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for moe_down_proj (production placement, default LoFi + fp32 DEST config).

Real rows only, against the fp64 reference on the dequantized bfloat8_b operands. Prints PCC, max / mean abs error,
relative RMS, bf16 ULP error, and the got/true ratio spread (scale-bug detector). Gates are the acceptance
tolerances.
"""

import torch
import ttnn

import pytest

# tests.ttnn.utils_for_testing (assert_with_pcc) does not import in this checkout; same metrics module instead.
from models.common.metrics import comp_allclose, compute_pcc
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import (
    LAYER0,
    LAYER3,
    TOLERANCES,
    check_counts,
    make_operands,
    reference,
    relative_rms,
    run,
)

LAYER5 = (228, 481, 3027, 332, 1944, 427, 1339, 414)

CASES = [
    ("t37", 37, (20, 0, 9, 14, 1, 11, 7, 12)),
    ("t1024_scaled", 1024, (605, 134, 384, 67, 75, 82, 311, 390)),
    ("t4096_layer0", 4096, LAYER0),
    ("t4096_layer3", 4096, LAYER3),
    ("t4096_layer5", 4096, LAYER5),
]


def bf16_ulp(values):
    """Spacing of bf16 at |values| (8 significand bits)."""
    exponent = torch.floor(torch.log2(values.abs().clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.pow(2.0, exponent - 7)


# LoFi (the default) truncates the w mantissa, which biases |y| low by about 1.35%; the scale-bug check therefore
# runs on HiFi2, where the ratio median must sit at 1.0.
FIDELITIES = [
    ("lofi_default", None, 0.03),
    ("hifi2", ttnn.MathFidelity.HiFi2, 1e-3),
]


@pytest.mark.parametrize("fidelity_id, fidelity, ratio_tol", FIDELITIES, ids=[f[0] for f in FIDELITIES])
@pytest.mark.parametrize("name, tokens, counts", CASES, ids=[case[0] for case in CASES])
def test_precision_baseline(device, name, tokens, counts, fidelity_id, fidelity, ratio_tol):
    check_counts(tokens, counts)
    config = None
    if fidelity is not None:
        config = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts)
    out = run(x_dev, w_dev, counts_dev, compute_kernel_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
    rows = xq.shape[-2]
    actual = ttnn.to_torch(out).reshape(rows, -1)
    # Free device tensors before any assert: a failure's traceback would otherwise pin them in L1.
    del out, x_dev, w_dev, counts_dev

    expected, real = reference(xq, wq, counts)
    actual = actual[real].double()
    expected = expected[real]

    min_pcc, max_rms = TOLERANCES[ttnn.bfloat8_b]
    got_pcc = float(compute_pcc(actual, expected))
    _, allclose_msg = comp_allclose(expected, actual)
    abs_err = (actual - expected).abs()
    rms = relative_rms(actual, expected)
    ulp = (abs_err / bf16_ulp(expected)).float()
    # Ratio over elements well above the noise floor; a tight cluster off 1.0 is a uniform scale bug.
    significant = expected.abs() > 0.1 * expected.abs().max()
    ratio = (actual[significant] / expected[significant]).float()
    p5, p50, p95 = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95])).tolist()

    print(
        f"\nPRECISION {name} {fidelity_id}: rows={int(real.sum())} pcc={got_pcc:.6f} max_abs={abs_err.max():.4e} "
        f"mean_abs={abs_err.mean():.4e} rel_rms={rms:.4e} ulp_p50={ulp.median():.1f} "
        f"ulp_p99={torch.quantile(ulp, 0.99):.1f} ratio_p5/p50/p95={p5:.4f}/{p50:.4f}/{p95:.4f} | {allclose_msg}"
    )
    assert got_pcc >= min_pcc, f"pcc {got_pcc:.6f} < {min_pcc}"
    assert rms <= max_rms, f"relative rms {rms:.4e} > {max_rms}"
    assert abs(p50 - 1.0) < ratio_tol, f"ratio median {p50:.4f} off 1.0 by more than {ratio_tol}"
