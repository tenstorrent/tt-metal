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


# ---------------------------------------------------------------------------
# Per-element error diagnostic. Used to determine whether ttnn.var error in the
# Welford path is uniform across outputs (per-element math issue) or localized
# (state-leak / padding issue).
# ---------------------------------------------------------------------------


def _per_element_ulp(device, op_name, shape, dim, correction, keepdim):
    torch.manual_seed(0)
    torch_input_fp32 = torch.randn(shape, dtype=torch.float32) + _INPUT_OFFSET
    torch_op = getattr(torch, op_name)
    ttnn_op = getattr(ttnn, op_name)
    expected_fp64 = torch_op(torch_input_fp32.to(torch.float64), dim=dim, keepdim=keepdim, correction=correction)

    tt_in = ttnn.from_torch(torch_input_fp32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn_op(tt_in, dim=dim, keepdim=keepdim, correction=correction, use_legacy=False)
    tt_out = ttnn.to_layout(ttnn.from_device(tt_out), ttnn.TILE_LAYOUT)
    actual_fp32 = ttnn.to_torch(tt_out)

    expected_d = expected_fp64.to(torch.float64)
    actual_d = actual_fp32.to(torch.float64).reshape(expected_d.shape)
    abs_err = torch.abs(actual_d - expected_d)
    expected_f = expected_fp64.to(torch.float32)
    ulp_at_expected = compute_ulp(expected_f).to(torch.float64)
    return (abs_err / ulp_at_expected.clamp(min=1e-300)).flatten().tolist(), expected_f.flatten().tolist()


def _summarize_distribution(values: list[float]) -> str:
    if not values:
        return "(empty)"
    t = torch.tensor(values, dtype=torch.float64)
    return (
        f"n={t.numel()} min={t.min().item():.1f} median={t.median().item():.1f} "
        f"mean={t.mean().item():.1f} max={t.max().item():.1f} "
        f"p90={t.quantile(0.90).item():.1f} p99={t.quantile(0.99).item():.1f}"
    )


def test_var_fp32_known_answer_diagnostic(device):
    """Run var on inputs with mathematically-exact answers so any ttnn output deviation is precision loss.

    Uses inputs that are exactly representable in fp32 with simple known variance:
        - Each row of 32 elements is the integer sequence [k*offset, k*offset+1, ..., k*offset+31]
          for some k. Variance of consecutive integers 0..N-1 is exactly (N^2-1)/12; with N=32
          that's (1024-1)/12 = 1023/12 = 85.25. Sample variance (correction=true) = 32*85.25/31 = 88.0.
        - We try 4 variants: small magnitudes (k=0), offset 100, offset 10000, offset 1e6.
          Larger offsets stress the cancellation regime; small offsets are near-noise-free.
    """
    N = 32
    expected_pop_var = (N * N - 1) / 12.0  # 85.25
    expected_sample_var = N * expected_pop_var / (N - 1)  # 88.0

    cases = [
        ("offset=0     (no cancellation)", 0.0),
        ("offset=100   (κ ≈ 11)", 100.0),
        ("offset=10000 (κ ≈ 1100)", 10000.0),
        ("offset=1e6   (κ ≈ 110000)", 1e6),
    ]

    # Run twice: once on dim=-1 (W-reduce, has transpose), once on dim=-2 (H-reduce, no transpose).
    for reduce_dim in (-1, -2):
        path = "W-reduce (with transpose)" if reduce_dim == -1 else "H-reduce (no transpose)"
        print(f"\n========== {path} ==========")
        for label, offset in cases:
            # 32 rows × 32 cols. For dim=-1: each row = [offset, offset+1, ..., offset+31].
            # For dim=-2: each col = [offset, offset+1, ..., offset+31] so we transpose the input.
            base = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(N, N).contiguous() + offset
            if reduce_dim == -2:
                base = base.transpose(-1, -2).contiguous()
            torch_input = base.unsqueeze(0).unsqueeze(0)

            tt_in = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            tt_out = ttnn.var(tt_in, dim=reduce_dim, keepdim=True, correction=True, use_legacy=False)
            actual = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(tt_out), ttnn.TILE_LAYOUT))

        actual_flat = actual.flatten().to(torch.float64)
        deviation = (actual_flat - expected_sample_var).abs()
        rel = deviation / expected_sample_var

        # ULP at the expected value (88.0) in fp32
        ulp_at_88 = compute_ulp(torch.tensor(expected_sample_var, dtype=torch.float32)).item()
        ulp_err = deviation / ulp_at_88

        print(f"\n--- {label} ---")
        print(f"  expected sample variance: {expected_sample_var}")
        print(f"  actual values: min={actual_flat.min().item():.6f} max={actual_flat.max().item():.6f}")
        print(
            f"  abs deviation: min={deviation.min().item():.4e} median={deviation.median().item():.4e} "
            f"max={deviation.max().item():.4e}"
        )
        print(
            f"  rel deviation: min={rel.min().item():.4e} median={rel.median().item():.4e} "
            f"max={rel.max().item():.4e}"
        )
        print(
            f"  ULP error: min={ulp_err.min().item():.1f} median={ulp_err.median().item():.1f} "
            f"max={ulp_err.max().item():.1f}"
        )


def test_var_fp32_per_element_diagnostic(device):
    """Print per-element ULP error distribution for several diagnostic shapes."""
    cases = [
        # (shape, dim, label)
        ((1, 1, 32, 32), -1, "W=32 H=32 (no padding, single tile)"),
        ((1, 1, 32, 20), -1, "W=20 H=32 (W has 12 cols of padding)"),
        ((1, 1, 32, 64), -1, "W=64 H=32 (2 W-tiles, no padding)"),
        ((1, 1, 64, 32), -1, "W=32 H=64 (2 H-tiles, tests state across NCHt)"),
        ((1, 1, 32, 32), -2, "H=32 W=32 dim=-2 (H-reduce, no padding)"),
        ((1, 1, 20, 32), -2, "H=20 W=32 dim=-2 (H has 12 rows of padding)"),
    ]

    for shape, dim, label in cases:
        ulps, expected_vals = _per_element_ulp(device, "var", shape, dim, correction=True, keepdim=True)
        print(f"\n--- {label} ---")
        print(f"shape={shape} dim={dim}  {len(ulps)} outputs")
        print(f"ULP error distribution: {_summarize_distribution(ulps)}")

        # Print the per-output ULP errors arranged as the output grid (excluding singleton dims).
        # For dim=-1: outputs along H. For dim=-2: outputs along W.
        out_dim_size = shape[-2] if dim == -1 else shape[-1]
        if len(ulps) == out_dim_size:
            # 1D output along the kept dimension
            lines = ["  per-output ULP error (index : ULP):"]
            for i, u in enumerate(ulps):
                marker = "  <-- worst" if u == max(ulps) else ""
                lines.append(f"    {i:3d} : {u:10.1f}{marker}")
            # Print only first/last 20 if too long
            if len(lines) > 25:
                print("\n".join(lines[:11] + ["    ..."] + lines[-12:]))
            else:
                print("\n".join(lines))
