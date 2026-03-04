#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate and verify polynomial coefficients for forward GELU piecewise approximation.

Strategy: Approximate Phi(x) = CDF of N(0,1) = 0.5*(1+erf(x/sqrt(2))) via polynomial,
then GELU(x) = x * Phi(x). This ensures GELU(0) = 0 exactly and handles the
linear growth naturally.

Regions:
1. x <= -13.1875: GELU(x) = 0 (BF16 saturation, verified by research)
2. -13.1875 < x <= -5: exp-based asymptotic formula
3. -5 < x <= -3: left shifted polynomial for GELU(x) directly
4. -3 < x < 3: core polynomial for Phi(x), then multiply by x
5. 3 <= x < 5.375: try identity or right polynomial
6. x >= 5.375: GELU(x) = x (BF16 identity saturation, verified by research)

DAZ+FTZ Model: Per Tenstorrent hardware, all denormals are treated as zero.
The golden reference function applies this model.

Run: python tests/ttnn/unit_tests/operations/eltwise/generate_gelu_fw_coefficients.py
"""

import struct
import math
import numpy as np

try:
    import mpmath

    mpmath.mp.dps = 80
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


# =============================================================================
# BF16 utilities with DAZ+FTZ model
# =============================================================================


def float_to_bf16_bits(f):
    f32_bytes = struct.pack(">f", f)
    return struct.unpack(">H", f32_bytes[:2])[0]


def bf16_bits_to_float(bits):
    f32_bytes = struct.pack(">H", bits) + b"\x00\x00"
    return struct.unpack(">f", f32_bytes)[0]


def is_bf16_denormal(bits):
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return exp == 0 and mantissa != 0


def round_to_bf16_daz_ftz(f):
    """Round to BF16 with DAZ+FTZ: denormals flush to zero."""
    bits = float_to_bf16_bits(f)
    if is_bf16_denormal(bits):
        return 0.0
    return bf16_bits_to_float(bits)


def apply_daz(f):
    """Denormals-Are-Zero: if input is denormal in BF16, read as zero."""
    bits = float_to_bf16_bits(f)
    if is_bf16_denormal(bits):
        return 0.0
    return f


def ulp_distance_bf16_daz(a, b):
    """ULP distance with DAZ model."""
    a_bits = float_to_bf16_bits(a)
    b_bits = float_to_bf16_bits(b)
    if is_bf16_denormal(a_bits):
        a_bits = 0x0000 if (a_bits & 0x8000) == 0 else 0x8000
    if is_bf16_denormal(b_bits):
        b_bits = 0x0000 if (b_bits & 0x8000) == 0 else 0x8000
    a_is_zero = (a_bits & 0x7FFF) == 0
    b_is_zero = (b_bits & 0x7FFF) == 0
    if a_is_zero and b_is_zero:
        return 0

    def to_linear(bits):
        if (bits & 0x7FFF) == 0:
            return 0
        if bits & 0x8000:
            return -(bits & 0x7FFF)
        return bits & 0x7FFF

    return abs(to_linear(a_bits) - to_linear(b_bits))


def all_bf16_values():
    """All valid BF16 values (no NaN, no Inf, no denormals)."""
    values = []
    for bits in range(0x10000):
        exp = (bits >> 7) & 0xFF
        mantissa = bits & 0x7F
        if exp == 0xFF:
            continue
        if exp == 0 and mantissa != 0:
            continue
        f = bf16_bits_to_float(bits)
        values.append(f)
    return sorted(set(values))


# =============================================================================
# High-precision reference with DAZ+FTZ model
# =============================================================================


def gelu_reference(x):
    """GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2))) with mpmath precision."""
    if HAS_MPMATH:
        xm = mpmath.mpf(x)
        return float(xm * mpmath.mpf("0.5") * (1 + mpmath.erf(xm / mpmath.sqrt(2))))
    if x >= 0:
        return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return x * 0.5 * math.erfc(abs(x) / math.sqrt(2.0))


def cdf_reference(x):
    """Phi(x) = 0.5*(1+erf(x/sqrt(2)))."""
    if HAS_MPMATH:
        xm = mpmath.mpf(x)
        return float(mpmath.mpf("0.5") * (1 + mpmath.erf(xm / mpmath.sqrt(2))))
    if x >= 0:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return 0.5 * math.erfc(abs(x) / math.sqrt(2.0))


def gelu_expected_bf16(x):
    """Golden reference: GELU(x) with DAZ input and FTZ output."""
    x_daz = apply_daz(x)
    result = gelu_reference(x_daz)
    return round_to_bf16_daz_ftz(result)


# =============================================================================
# Polynomial utilities
# =============================================================================


def chebyshev_nodes(n, a, b):
    k = np.arange(1, n + 1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))


def eval_poly_horner(coeffs, x):
    """Evaluate polynomial. coeffs = [c0, c1, c2, ...] i.e. c0 + c1*x + c2*x² + ..."""
    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[i]
    return result


def eval_poly_horner_f32(coeffs, x):
    """Evaluate polynomial in float32 (simulating hardware)."""
    result = np.float32(coeffs[-1])
    for i in range(len(coeffs) - 2, -1, -1):
        result = np.float32(np.float32(result) * np.float32(x) + np.float32(coeffs[i]))
    return float(result)


# =============================================================================
# Approach 1: CDF polynomial, then GELU(x) = x * Phi(x)
# =============================================================================


def fit_cdf_polynomial(degree, a, b, n_oversample=3):
    """Fit Phi(x) on [a, b] with Chebyshev interpolation."""
    n_nodes = degree * n_oversample
    nodes = chebyshev_nodes(n_nodes, a, b)
    values = np.array([cdf_reference(float(x)) for x in nodes])
    coeffs = np.polyfit(nodes, values, degree)[::-1]  # c0, c1, ...
    return coeffs


def verify_cdf_approach(core_coeffs, left_coeffs, right_coeffs=None):
    """Verify CDF approach: GELU(x) = x * Phi_poly(x)."""
    all_vals = all_bf16_values()
    INV_SQRT_2PI = 0.3989422804014327

    regions = {}
    for name in ["zero_sat", "exp_based", "left_poly", "core_poly", "right_poly", "identity"]:
        regions[name] = {"count": 0, "max_ulp": 0, "worst_x": 0, "ulp_sum": 0, "ulp_hist": {}}

    for x in all_vals:
        expected = gelu_expected_bf16(x)

        if x >= 5.375:
            region = "identity"
            result = x
        elif x >= 3.0:
            if right_coeffs is not None:
                region = "right_poly"
                phi = eval_poly_horner(right_coeffs, x)
                result = x * phi
            else:
                region = "right_poly"
                result = x  # identity fallback
        elif x >= -3.0:
            region = "core_poly"
            phi = eval_poly_horner(core_coeffs, x)
            result = x * phi
        elif x >= -5.0:
            region = "left_poly"
            t = x + 4.0
            phi = eval_poly_horner(left_coeffs, t)
            result = x * phi
        elif x > -13.1875:
            region = "exp_based"
            x2 = x * x
            t_val = x2 * (-0.5)
            exp_val = math.exp(t_val)
            inv_x2 = 1.0 / x2
            inv_x4 = inv_x2 * inv_x2
            correction = 1.0 - inv_x2 + 3.0 * inv_x4
            result = -exp_val * INV_SQRT_2PI * correction
        else:
            region = "zero_sat"
            result = 0.0

        actual = round_to_bf16_daz_ftz(result)
        ulp = ulp_distance_bf16_daz(actual, expected)

        r = regions[region]
        r["count"] += 1
        r["ulp_sum"] += ulp
        if ulp > r["max_ulp"]:
            r["max_ulp"] = ulp
            r["worst_x"] = x
        r["ulp_hist"][ulp] = r["ulp_hist"].get(ulp, 0) + 1

    return regions


def approach_cdf():
    """Main approach: CDF polynomial."""
    print("=" * 70)
    print("APPROACH: CDF Polynomial Phi(x), GELU(x) = x * Phi(x)")
    print("=" * 70)

    # Core CDF [-3, 3], degree 16
    print("\n--- Core CDF [-3, 3]: Degree 16 ---")
    core_cdf = fit_cdf_polynomial(16, -3.0, 3.0)
    print("Phi(x) coefficients (c0 + c1*x + ...):")
    for i, c in enumerate(core_cdf):
        print(f"  c{i:2d} = {c:.15e}")

    # Left CDF [-5, -3], shifted t = x+4, degree 8
    print("\n--- Left CDF [-5, -3]: Degree 8, shifted t=x+4 ---")
    # Fit Phi(x) where x = t - 4, t ∈ [-1, 1]
    left_nodes = chebyshev_nodes(24, -5.0, -3.0)
    left_cdf_values = np.array([cdf_reference(float(x)) for x in left_nodes])
    left_nodes_shifted = left_nodes + 4.0
    left_cdf = np.polyfit(left_nodes_shifted, left_cdf_values, 8)[::-1]
    print("Phi(t) coefficients where t = x+4:")
    for i, c in enumerate(left_cdf):
        print(f"  c{i:2d} = {c:.15e}")

    # Right CDF [3, 5.375], degree 8
    print("\n--- Right CDF [3, 5.375]: Degree 8 ---")
    right_cdf = fit_cdf_polynomial(8, 3.0, 5.375)
    print("Phi(x) coefficients:")
    for i, c in enumerate(right_cdf):
        print(f"  c{i:2d} = {c:.15e}")

    # Verify
    regions = verify_cdf_approach(core_cdf, left_cdf, right_cdf)

    print(f"\n{'Region':>15s} {'Count':>7s} {'Max ULP':>8s} {'Mean ULP':>10s} {'Worst x':>12s}")
    print("-" * 60)
    overall_max = 0
    for name, r in regions.items():
        if r["count"] > 0:
            mean = r["ulp_sum"] / r["count"]
            print(f"{name:>15s} {r['count']:>7d} {r['max_ulp']:>8d} {mean:>10.4f} {r['worst_x']:>12.6f}")
            if r["max_ulp"] > overall_max:
                overall_max = r["max_ulp"]
    print(f"\nOverall Max ULP: {overall_max}")

    # Also try without right polynomial (identity for x >= 3)
    print("\n--- Without right polynomial (identity for x >= 3) ---")
    regions2 = verify_cdf_approach(core_cdf, left_cdf, right_coeffs=None)
    for name, r in regions2.items():
        if r["count"] > 0 and name == "right_poly":
            print(f"  Identity [3, 5.375): Max ULP = {r['max_ulp']}, count = {r['count']}")

    return core_cdf, left_cdf, right_cdf


# =============================================================================
# Approach 2: Direct GELU(x) polynomial (skip CDF multiply)
# =============================================================================


def approach_direct():
    """Fit GELU(x) directly, but ensure p(0) = 0 by constraining c0=0."""
    print("\n" + "=" * 70)
    print("APPROACH: Direct GELU(x) with constrained p(0)=0")
    print("=" * 70)

    # Core [-3, 3]: fit p(x) = c1*x + c2*x² + ... (no constant term)
    # GELU(x) = x * g(x) where g(x) ≈ 0.5 near 0
    # Equivalently: fit g(x) = GELU(x)/x (= Phi(x)) and output x*g(x)
    # This is the same as the CDF approach!
    print("Direct GELU(x) with p(0)=0 is equivalent to CDF approach.")
    print("Skipping (use CDF approach instead).")


# =============================================================================
# Approach 3: Wider core polynomial [-5, 5.375]
# =============================================================================


def approach_wide_cdf():
    """Try a single wide CDF polynomial covering [-5, 5.375]."""
    print("\n" + "=" * 70)
    print("APPROACH: Wide CDF Polynomial [-5, 5.375], Degree 20")
    print("=" * 70)

    wide_cdf = fit_cdf_polynomial(20, -5.0, 5.375, n_oversample=4)
    print("Wide Phi(x) coefficients:")
    for i, c in enumerate(wide_cdf):
        print(f"  c{i:2d} = {c:.15e}")

    # Verify using the wide polynomial for everything in [-5, 5.375]
    all_vals = all_bf16_values()
    INV_SQRT_2PI = 0.3989422804014327

    max_ulp = 0
    worst_x = 0
    ulp_hist = {}
    count = 0

    for x in all_vals:
        expected = gelu_expected_bf16(x)

        if x >= 5.375:
            result = x
        elif x >= -5.0:
            phi = eval_poly_horner(wide_cdf, x)
            result = x * phi
        elif x > -13.1875:
            x2 = x * x
            t_val = x2 * (-0.5)
            exp_val = math.exp(t_val)
            inv_x2 = 1.0 / x2
            inv_x4 = inv_x2 * inv_x2
            correction = 1.0 - inv_x2 + 3.0 * inv_x4
            result = -exp_val * INV_SQRT_2PI * correction
        else:
            result = 0.0

        actual = round_to_bf16_daz_ftz(result)
        ulp = ulp_distance_bf16_daz(actual, expected)
        count += 1
        ulp_hist[ulp] = ulp_hist.get(ulp, 0) + 1
        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x

    print(f"\nWide CDF verification ({count} BF16 values):")
    print(f"  Max ULP: {max_ulp} (at x={worst_x})")
    top_ulps = sorted(ulp_hist.items())[:10]
    print(f"  ULP histogram (top 10): {dict(top_ulps)}")


# =============================================================================
# Approach 4: GELU(x)/x = Phi(x) on core, GELU(x) direct on sides
# =============================================================================


def approach_mixed():
    """CDF on core [-3, 3], direct GELU on left [-5, -3] and right [3, 5.375]."""
    print("\n" + "=" * 70)
    print("APPROACH: Mixed — CDF core + direct GELU sides")
    print("=" * 70)

    # Core: CDF degree 16 on [-3, 3]
    core_cdf = fit_cdf_polynomial(16, -3.0, 3.0)

    # Left: direct GELU on [-5, -3], shifted t = x+4
    left_nodes = chebyshev_nodes(24, -5.0, -3.0)
    left_gelu_values = np.array([gelu_reference(float(x)) for x in left_nodes])
    left_shifted = left_nodes + 4.0
    left_gelu = np.polyfit(left_shifted, left_gelu_values, 8)[::-1]

    # Right: direct GELU on [3, 5.375]
    right_nodes = chebyshev_nodes(24, 3.0, 5.375)
    right_gelu_values = np.array([gelu_reference(float(x)) for x in right_nodes])
    right_gelu = np.polyfit(right_nodes, right_gelu_values, 8)[::-1]

    # Verify
    all_vals = all_bf16_values()
    INV_SQRT_2PI = 0.3989422804014327

    regions = {}
    for name in ["zero_sat", "exp_based", "left_direct", "core_cdf", "right_direct", "identity"]:
        regions[name] = {"count": 0, "max_ulp": 0, "worst_x": 0, "ulp_sum": 0}

    for x in all_vals:
        expected = gelu_expected_bf16(x)

        if x >= 5.375:
            region = "identity"
            result = x
        elif x >= 3.0:
            region = "right_direct"
            result = eval_poly_horner(right_gelu, x)
        elif x >= -3.0:
            region = "core_cdf"
            phi = eval_poly_horner(core_cdf, x)
            result = x * phi
        elif x >= -5.0:
            region = "left_direct"
            t = x + 4.0
            result = eval_poly_horner(left_gelu, t)
        elif x > -13.1875:
            region = "exp_based"
            x2 = x * x
            t_val = x2 * (-0.5)
            exp_val = math.exp(t_val)
            inv_x2 = 1.0 / x2
            inv_x4 = inv_x2 * inv_x2
            correction = 1.0 - inv_x2 + 3.0 * inv_x4
            result = -exp_val * INV_SQRT_2PI * correction
        else:
            region = "zero_sat"
            result = 0.0

        actual = round_to_bf16_daz_ftz(result)
        ulp = ulp_distance_bf16_daz(actual, expected)

        r = regions[region]
        r["count"] += 1
        r["ulp_sum"] += ulp
        if ulp > r["max_ulp"]:
            r["max_ulp"] = ulp
            r["worst_x"] = x

    print(f"\n{'Region':>15s} {'Count':>7s} {'Max ULP':>8s} {'Mean ULP':>10s} {'Worst x':>12s}")
    print("-" * 60)
    overall_max = 0
    for name, r in regions.items():
        if r["count"] > 0:
            mean = r["ulp_sum"] / r["count"]
            print(f"{name:>15s} {r['count']:>7d} {r['max_ulp']:>8d} {mean:>10.4f} {r['worst_x']:>12.6f}")
            if r["max_ulp"] > overall_max:
                overall_max = r["max_ulp"]
    print(f"\nOverall Max ULP: {overall_max}")

    return core_cdf, left_gelu, right_gelu


if __name__ == "__main__":
    print("Forward GELU Coefficient Generation and Verification")
    print("Saturation thresholds: zero at x <= -13.1875, identity at x >= 5.375")
    print("DAZ+FTZ model applied to all reference values")
    print()

    # Try all approaches
    core_cdf, left_cdf, right_cdf = approach_cdf()
    approach_direct()
    approach_wide_cdf()
    core_cdf_mixed, left_gelu_mixed, right_gelu_mixed = approach_mixed()

    # Output best coefficients
    print("\n" + "=" * 70)
    print("BEST COEFFICIENTS FOR KERNEL")
    print("=" * 70)
    print("\n// Core CDF [-3, 3], degree 16")
    print("// GELU(x) = x * Phi(x) where Phi(x) = polynomial(x)")
    for i, c in enumerate(core_cdf):
        print(f"constexpr float GELU_CDF_CORE_C{i} = {c:.15e}f;")

    print("\n// Left CDF [-5, -3], degree 8, shifted t = x + 4")
    print("// Phi(x) = polynomial(x + 4)")
    for i, c in enumerate(left_cdf):
        print(f"constexpr float GELU_CDF_LEFT_C{i} = {c:.15e}f;")

    print("\n// Right CDF [3, 5.375], degree 8")
    print("// Phi(x) = polynomial(x)")
    for i, c in enumerate(right_cdf):
        print(f"constexpr float GELU_CDF_RIGHT_C{i} = {c:.15e}f;")
