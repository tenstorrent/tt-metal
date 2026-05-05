# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Measures worst-case ULP, relative, absolute, and Frobenius errors for ttnn.std and ttnn.var
in float32. Covers the same shapes and dims exercised by test_std / test_var in test_reduction.py
plus a large 5D shape (3, 6, 40, 63, 20).

Reference is computed in fp64 (input cast to fp64 before torch.std/torch.var) so the reference
keeps precision below 1 fp32 ULP. ULP is reported in fp32 units.

No accuracy thresholds are enforced — run with -s to see the printed tables.

Usage:
    source python_env/bin/activate && pytest \
        tests/ttnn/unit_tests/operations/reduce/test_std_var_fp32_errors.py -s
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import ulp as compute_ulp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure_errors(expected_fp64: torch.Tensor, actual_fp32: torch.Tensor) -> dict:
    """Compare an fp32 actual against an fp64 reference. ULP is reported in fp32 units."""
    expected_d = expected_fp64.to(torch.float64)
    actual_d = actual_fp32.to(torch.float64).reshape(expected_d.shape)

    abs_err = torch.abs(actual_d - expected_d)
    max_abs = abs_err.max().item()

    denom = torch.abs(expected_d)
    rel_err = torch.where(denom > 0, abs_err / denom.clamp(min=1e-300), torch.zeros_like(abs_err))
    max_rel = rel_err.max().item()

    error = actual_d - expected_d
    frob_error = torch.norm(error, p="fro").item()
    frob_expected = torch.norm(expected_d, p="fro").item()
    frob = frob_error / frob_expected if frob_expected > 0 else frob_error

    # fp32 ULP at the expected value, with the diff computed in fp64 so we don't
    # lose precision below 1 fp32 ULP.
    expected_f = expected_fp64.to(torch.float32)
    ulp_at_expected = compute_ulp(expected_f).to(torch.float64)

    abs_v = torch.abs(expected_f.to(torch.float64))
    expected_max = abs_v.max().item()
    if expected_max > 0:
        normal_mask = abs_v >= 0.01 * expected_max
        if normal_mask.any():
            max_ulp = (abs_err[normal_mask] / ulp_at_expected[normal_mask]).max().item()
        else:
            max_ulp = 0.0
    else:
        max_ulp = 0.0

    return {"max_abs": max_abs, "max_rel": max_rel, "frobenius": frob, "max_ulp": max_ulp}


_INPUT_OFFSET = 50.0


def _run_op(device, op_name, shape, dim, correction, keepdim):
    torch.manual_seed(0)
    # Offset shifts the mean away from zero so variance computation hits the
    # subtraction-of-similar-values regime: (x - M) in Welford loses precision
    # when |mean| is large relative to the spread.
    torch_input_fp32 = torch.randn(shape, dtype=torch.float32) + _INPUT_OFFSET
    torch_op = getattr(torch, op_name)
    ttnn_op = getattr(ttnn, op_name)
    # Reference in fp64. torch's std/var on fp32 input already accumulates
    # internally in fp64, but the result is rounded back to fp32 — feeding fp64
    # input keeps that precision in the reference.
    torch_input_fp64 = torch_input_fp32.to(torch.float64)
    torch_ref_fp64 = torch_op(torch_input_fp64, dim=dim, keepdim=keepdim, correction=correction)

    tt_in = ttnn.from_torch(torch_input_fp32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn_op(tt_in, dim=dim, keepdim=keepdim, correction=correction, use_legacy=False)
    tt_out = ttnn.to_layout(ttnn.from_device(tt_out), ttnn.TILE_LAYOUT)
    actual = ttnn.to_torch(tt_out)

    return _measure_errors(torch_ref_fp64, actual)


def _print_table(rows: list[tuple], header: str):
    col_w = [max(len(str(r[i])) for r in [rows[0]] + rows[1:]) for i in range(len(rows[0]))]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(f"\n{header}")
    print(sep)
    print(fmt.format(*rows[0]))
    print(sep)
    for row in rows[1:]:
        print(fmt.format(*row))
    print(sep)


def _run_cases(device, op_name: str, cases: list[tuple]):
    header = ("shape", "dim", "correction", "keepdim", "max_abs", "max_rel", "frobenius", "max_ulp")
    rows = [header]
    worst = {"max_abs": 0.0, "max_rel": 0.0, "frobenius": 0.0, "max_ulp": 0.0}

    for shape, dim, correction, keepdim in cases:
        m = _run_op(device, op_name, shape, dim, correction, keepdim)
        rows.append(
            (
                str(tuple(shape)),
                str(dim),
                correction,
                keepdim,
                f"{m['max_abs']:.4e}",
                f"{m['max_rel']:.4e}",
                f"{m['frobenius']:.4e}",
                f"{m['max_ulp']:.1f}",
            )
        )
        for k in worst:
            worst[k] = max(worst[k], m[k])

    _print_table(rows, f"ttnn.{op_name} — float32 error measurement (fp64 reference)")
    rows_worst = [
        ("metric", "worst_case"),
        ("max_abs", f"{worst['max_abs']:.4e}"),
        ("max_rel", f"{worst['max_rel']:.4e}"),
        ("frobenius", f"{worst['frobenius']:.4e}"),
        ("max_ulp", f"{worst['max_ulp']:.1f}"),
    ]
    _print_table(rows_worst, f"ttnn.{op_name} — float32 worst-case summary")


# ---------------------------------------------------------------------------
# Test cases — mirrors the shapes/dims from test_std and test_var, plus a
# large 5D shape (3, 6, 40, 63, 20).
# ---------------------------------------------------------------------------

_LARGE_5D = (3, 6, 40, 63, 20)
_LARGE_5D_DIMS = [None, -1, -2, (-2, -1)]

_STD_CASES = [
    # (shape, dim, correction, keepdim)
    ((1, 32, 32), -1, True, True),
    ((1, 32, 32), -1, False, True),
    ((1, 32, 32), -2, True, True),
    ((1, 32, 32), -2, False, True),
    ((1, 32, 32), 0, True, True),
    ((1, 32, 32), 0, False, True),
    ((1, 32, 32), (-2, -1), True, True),
    ((1, 32, 32), (-2, -1), False, True),
    ((1, 32, 32), None, True, True),
    ((1, 32, 32), None, False, True),
    ((16, 64, 64), -1, True, True),
    ((16, 64, 64), -2, True, True),
    ((16, 64, 64), (-2, -1), True, True),
    ((16, 64, 64), None, True, True),
    ((16, 64, 64), -1, True, False),
    ((16, 64, 64), -2, True, False),
    ((16, 64, 64), None, True, False),
] + [(_LARGE_5D, dim, True, True) for dim in _LARGE_5D_DIMS]

_VAR_CASES = [
    # (shape, dim, correction, keepdim)
    ((1, 32, 32), -1, True, True),
    ((1, 32, 32), -1, False, True),
    ((1, 32, 32), -2, True, True),
    ((1, 32, 32), -2, False, True),
    ((1, 32, 32), (-2, -1), True, True),
    ((1, 32, 32), (-2, -1), False, True),
    ((1, 32, 32), None, True, True),
    ((1, 32, 32), None, False, True),
    ((1, 32, 32), [], True, True),
    ((16, 64, 64), -1, True, True),
    ((16, 64, 64), -2, True, True),
    ((16, 64, 64), (-2, -1), True, True),
    ((16, 64, 64), None, True, True),
    ((16, 64, 64), -1, False, True),
    ((16, 64, 64), None, False, True),
] + [(_LARGE_5D, dim, True, True) for dim in _LARGE_5D_DIMS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_std_fp32_errors(device):
    """Measure worst-case errors for ttnn.std with float32 input vs fp64 reference."""
    _run_cases(device, "std", _STD_CASES)


def test_var_fp32_errors(device):
    """Measure worst-case errors for ttnn.var with float32 input vs fp64 reference."""
    _run_cases(device, "var", _VAR_CASES)
