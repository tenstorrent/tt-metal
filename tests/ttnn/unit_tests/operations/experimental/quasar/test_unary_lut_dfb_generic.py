# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
GENERIC DFB eltwise flow validation (STEP 2/3): drive a representative subset of the
fitter's deployed activation configs through the unary_lut DFB op on craq-sim Quasar
and PCC-check vs the fitter ground_truth.

This is the DFB analog of tt-polynomial-fitter/quasar_sweep.sh: the same op + kernel,
parameterized over (activation, eval_method) by a `LutConfig` parsed from a read-only
fitter coefficient CSV. NO per-activation special-casing — the CSV (POLY vs RATIONAL,
degree, segments, coefficients) + the activation JSON (via ground_truth) drive
everything.

Coverage:
  * gelu       — POLY multi-segment (deg-5, 3-seg)
  * sigmoid    — POLY multi-segment (deg-5, 8-seg) AND POLY single-segment (deg-6)
  * exp        — POLY single-segment (deg-3)
  * atanh      — RATIONAL single-segment (n6d6 and n8d8)
  * abs        — trivial POLY (deg-1, 2-seg)

PCC is checked against BOTH the kernel's approximation (isolates DFB-path correctness)
and the TRUE activation (end-to-end fit+path), the latter being the headline metric per
the task. Each CSV is sampled over its OWN deployed [lo, hi] domain (no range reduction
in this DFB slice), so the true-activation PCC reflects the deployed config's fidelity
on its own fit domain.

Run on the Quasar simulator (see test_unary_lut_dfb.py header for the full env).
"""

import os
import sys
from pathlib import Path

import pytest

# Import the sibling driver module by directory (no package __init__ chain here).
sys.path.insert(0, str(Path(__file__).parent))
import dfb_lut_driver as drv  # noqa: E402

_COEFFS = Path(os.environ.get("TT_POLY_FITTER", "/localdev/nkapre/tt-polynomial-fitter")) / "data" / "coefficients"

_PCC = 0.99

# (csv filename, expected eval method tag) — the representative subset.
#
# Selection criterion: DFB-deployable WITHOUT range reduction (this slice has none), i.e.
# the deployed config's approximation is faithful to the true activation on its own fit
# domain. Full-domain sigmoid multi-segment configs (e.g. sigmoid_p6_s4 / p10_s3) are NOT
# in this set: they rely on range reduction / sign symmetry that the fitter applies
# outside the LUT, so their bare-LUT tails are wrong (approx-vs-true PCC ~0.0-0.46) — they
# are out of scope for the no-RR DFB slice, not a kernel defect. gelu provides the clean
# poly multi-segment full-domain case.
_CASES = [
    ("abs_p1_s2_chebyshev_any_mae.csv", "POLY"),  # trivial poly, deg-1 2-seg
    ("exp_p3_s1_uniform_fpminimax_ulp.csv", "POLY"),  # poly single-seg
    ("sigmoid_p6_s1_uniform_fpminimax_ulp.csv", "POLY"),  # poly single-seg
    ("gelu_p5_s3_uniform_any_ulp.csv", "POLY"),  # poly multi-seg (deg-5, 3-seg)
    ("gelu_p2_s8_uniform_any_ulp.csv", "POLY"),  # poly multi-seg (deg-2, 8-seg)
    ("atanh_n6d6_s1_uniform_rational_ulp.csv", "RATIONAL"),  # rational single-seg
    ("atanh_n8d8_s1_uniform_rational_ulp.csv", "RATIONAL"),  # rational single-seg
]


# Range-reduction (RR) cases — the activations that need reduce-then-poly-then-reconstruct
# to be correct over their FULL original domain (out of scope for the no-RR _CASES above).
# The kernel now performs RR uniformly, driven by the CSV's range_reduction_method METADATA
# (NO per-activation special-casing). (csv filename, expected rr_method code).
#   2 = exp (Cody-Waite), 7 = trig (sin/cos), 4 = exponent_alu_exp2 (exp/sigmoid),
#   5 = exponent_alu_log2 (log/log2/log10), 6 = exponent_alu_pow (sqrt/rsqrt/cbrt).
_RR_CASES = [
    ("exp_p5_s1_uniform_any_ulp.csv", 2),  # REDUCED_POLY: Cody-Waite exp(x)=2^k*exp(s)
    ("sin_p5_s1_uniform_any_ulp.csv", 7),  # REDUCED_POLY: trig reduce to [-pi/2,pi/2]
    ("cos_p6_s1_uniform_any_ulp.csv", 7),  # REDUCED_POLY: trig (even)
    ("exp_p5_s1_uniform_fpminimax_ulp.csv", 4),  # EXPONENT_ALU: exp2 exman/floor
    ("sigmoid_p6_s1_uniform_fpminimax_ulp.csv", 4),  # EXPONENT_ALU: exp2 + sigmoid compose
    ("log2_p4_s1_uniform_fpminimax_ulp.csv", 5),  # EXPONENT_ALU: log2 exexp + (m-1) basis
    ("log_p4_s1_uniform_fpminimax_ulp.csv", 5),  # EXPONENT_ALU: log2-basis * scale
    ("log10_p4_s1_uniform_fpminimax_ulp.csv", 5),  # EXPONENT_ALU: log2-basis * scale
    ("sqrt_p4_s1_uniform_fpminimax_ulp.csv", 6),  # NEWTON_ROOT analog: pow exexp/divmod (n=2)
    ("rsqrt_p4_s1_uniform_fpminimax_ulp.csv", 6),  # pow + reciprocal (n=2)
    ("cbrt_p3_s1_uniform_fpminimax_ulp.csv", 6),  # pow odd root (n=3, sign restore)
]


@pytest.mark.parametrize("csv_name,method_tag", _CASES, ids=[c[0] for c in _CASES])
def test_generic_dfb_activation(device, csv_name, method_tag):
    csv_path = _COEFFS / csv_name
    if not csv_path.exists():
        pytest.skip(f"CSV not found: {csv_path}")

    res = drv.run_dfb(device, str(csv_path), tiles=4)

    assert res["eval_method"] == method_tag, f"expected {method_tag}, parsed {res['eval_method']}"
    print(
        f"\n[{res['activation']:>8} {res['eval_method']:>8} deg={res['degree']} "
        f"seg={res['num_segments']} dom={res['domain']}]  "
        f"PCC_approx={res['pcc_vs_approx']:.6f}  PCC_true={res['pcc_vs_true']:.6f}"
    )
    # DFB-path correctness: the kernel must reproduce its own approximation tightly.
    assert res["pcc_vs_approx"] >= _PCC, f"DFB path PCC vs approximation {res['pcc_vs_approx']} < {_PCC}"
    # End-to-end accuracy vs the true activation on the deployed domain.
    assert res["pcc_vs_true"] >= _PCC, f"PCC vs ground_truth {res['pcc_vs_true']} < {_PCC}"


@pytest.mark.parametrize("csv_name,rr_code", _RR_CASES, ids=[c[0] for c in _RR_CASES])
def test_generic_dfb_range_reduction(device, csv_name, rr_code):
    """Range-reduction activations: reduce-then-poly-then-reconstruct over the FULL
    original domain. PCC is checked vs the TRUE activation (the kernel reconstructs it,
    so the reduced-poly approximation golden does not apply — run_dfb reports
    pcc_vs_approx == pcc_vs_true for RR)."""
    csv_path = _COEFFS / csv_name
    if not csv_path.exists():
        pytest.skip(f"CSV not found: {csv_path}")

    res = drv.run_dfb(device, str(csv_path), tiles=4)

    assert res["rr_enabled"], f"{csv_name}: expected range reduction enabled, got rr_method={res['rr_method']}"
    assert res["rr_method"] == rr_code, f"{csv_name}: expected rr_method {rr_code}, parsed {res['rr_method']}"
    print(
        f"\n[{res['activation']:>8} rr={res['rr_method']} {res['eval_method']:>8} "
        f"seg={res['num_segments']} dom={res['domain']}]  PCC_true={res['pcc_vs_true']:.6f}"
    )
    # End-to-end accuracy vs the true activation over the full original domain.
    assert res["pcc_vs_true"] >= _PCC, f"RR PCC vs ground_truth {res['pcc_vs_true']} < {_PCC}"
