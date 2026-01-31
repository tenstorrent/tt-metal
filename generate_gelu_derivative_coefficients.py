#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate polynomial coefficients for GELU'(x) = d/dx GELU(x) approximation.
Uses Remez exchange algorithm for minimax polynomial approximation.

GELU'(x) = cdf + x * pdf
where:
  cdf = 0.5 * (1 + erf(x / sqrt(2)))
  pdf = exp(-x²/2) / sqrt(2π)

For numerical stability with negative x, use erfc:
  cdf = 0.5 * erfc(-x / sqrt(2))  when x < 0
"""

import numpy as np
import struct
from typing import Tuple, List
import warnings

# Try to import mpmath for high-precision reference
try:
    from mpmath import mp, erf, erfc, exp, sqrt, pi

    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False
    print("Warning: mpmath not available, using numpy (less precise)")


def gelu_derivative_mpmath(x: float) -> float:
    """Compute GELU'(x) with 256-bit precision using mpmath."""
    if not HAVE_MPMATH:
        return gelu_derivative_numpy(x)

    mp.prec = 256
    x_mp = mp.mpf(str(x))
    sqrt2 = sqrt(mp.mpf(2))
    sqrt_2pi = sqrt(2 * pi)

    # Use erfc for numerical stability with negative x
    if x < 0:
        cdf = mp.mpf("0.5") * erfc(-x_mp / sqrt2)
    else:
        cdf = mp.mpf("0.5") * (1 + erf(x_mp / sqrt2))

    pdf = exp(-x_mp * x_mp / 2) / sqrt_2pi
    return float(cdf + x_mp * pdf)


def gelu_derivative_numpy(x: float) -> float:
    """Compute GELU'(x) using numpy (fp64)."""
    import math

    sqrt2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)

    if x < 0:
        cdf = 0.5 * math.erfc(-x / sqrt2)
    else:
        cdf = 0.5 * (1.0 + math.erf(x / sqrt2))

    pdf = math.exp(-0.5 * x * x) / sqrt_2pi
    return cdf + x * pdf


def gelu_derivative(x):
    """Compute GELU'(x) - vectorized version for arrays."""
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    result = np.zeros_like(x)
    for i, xi in enumerate(x):
        if HAVE_MPMATH:
            result[i] = gelu_derivative_mpmath(xi)
        else:
            result[i] = gelu_derivative_numpy(xi)

    if scalar_input:
        return result[0]
    return result


def float_to_bf16(f: float) -> int:
    """Convert float32 to bfloat16 bits."""
    bits = struct.unpack(">I", struct.pack(">f", float(f)))[0]
    return bits >> 16


def bf16_to_float(bf16_bits: int) -> float:
    """Convert bfloat16 bits to float32."""
    bits = int(bf16_bits) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial using Horner's method."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def fit_minimax_remez(
    func, degree: int, x_min: float, x_max: float, max_iters: int = 100, tol: float = 1e-12
) -> np.ndarray:
    """Remez exchange algorithm for minimax polynomial approximation."""
    n = degree + 1

    # Initial reference points (Chebyshev nodes)
    k = np.arange(n + 1)
    ref_points = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / n)
    ref_points = np.sort(ref_points)

    coeffs = None

    for iteration in range(max_iters):
        # Build system of equations
        A = np.zeros((n + 1, n + 1))
        b = np.array([func(x) for x in ref_points])

        for i, x in enumerate(ref_points):
            for j in range(n):
                A[i, j] = x**j
            A[i, n] = (-1) ** i

        try:
            sol = np.linalg.solve(A, b)
            coeffs = sol[:n]
        except np.linalg.LinAlgError:
            # Perturb points slightly and retry
            ref_points += np.random.randn(n + 1) * 1e-8
            continue

        # Find new reference points at error extrema
        x_dense = np.linspace(x_min, x_max, 20000)
        y_exact = np.array([func(x) for x in x_dense])
        y_poly = eval_poly(coeffs, x_dense)
        errors = y_exact - y_poly

        # Find local extrema
        extrema_indices = [0]
        for i in range(1, len(errors) - 1):
            if (errors[i - 1] < errors[i] > errors[i + 1]) or (errors[i - 1] > errors[i] < errors[i + 1]):
                extrema_indices.append(i)
        extrema_indices.append(len(errors) - 1)

        extrema_x = x_dense[extrema_indices]
        extrema_err = np.abs(errors[extrema_indices])

        # Select n+1 points with largest errors
        sorted_idx = np.argsort(-extrema_err)
        new_ref = np.sort(extrema_x[sorted_idx[: n + 1]])

        # Check convergence
        if np.max(np.abs(new_ref - ref_points)) < tol:
            break
        ref_points = new_ref

    return coeffs


def compute_ulp_error(expected: float, actual: float) -> int:
    """Compute ULP distance between two BF16 values."""
    if expected == 0 and actual == 0:
        return 0

    # Handle DAZ (denormals as zero)
    exp_bf16 = float_to_bf16(expected)
    act_bf16 = float_to_bf16(actual)

    # Normalize denormals to zero
    if (exp_bf16 & 0x7F80) == 0 and (exp_bf16 & 0x007F) != 0:
        exp_bf16 = 0
    if (act_bf16 & 0x7F80) == 0 and (act_bf16 & 0x007F) != 0:
        act_bf16 = 0

    if exp_bf16 == act_bf16:
        return 0

    def bf16_to_signed_int(bf16):
        if bf16 >= 0x8000:
            return -(0x10000 - bf16)
        return bf16

    return abs(bf16_to_signed_int(exp_bf16) - bf16_to_signed_int(act_bf16))


def analyze_polynomial(coeffs: np.ndarray, x_min: float, x_max: float, func, name: str = ""):
    """Analyze polynomial accuracy over a range."""
    x_test = np.linspace(x_min, x_max, 10000)
    y_exact = np.array([func(x) for x in x_test])
    y_poly = eval_poly(coeffs, x_test)

    # Compute ULP errors
    ulps = []
    for i in range(len(x_test)):
        ulps.append(compute_ulp_error(y_exact[i], y_poly[i]))
    ulps = np.array(ulps)

    max_ulp = np.max(ulps)
    mean_ulp = np.mean(ulps)
    pct_le_1 = 100.0 * np.mean(ulps <= 1)
    pct_le_2 = 100.0 * np.mean(ulps <= 2)

    worst_idx = np.argmax(ulps)
    worst_x = x_test[worst_idx]

    print(
        f"  [{x_min:6.2f}, {x_max:6.2f}]: Max ULP = {max_ulp:5d}, "
        f"Mean = {mean_ulp:6.2f}, ≤1 ULP: {pct_le_1:5.1f}%, "
        f"≤2 ULP: {pct_le_2:5.1f}%, Worst x = {worst_x:.4f}"
    )

    return max_ulp, mean_ulp, worst_x


def main():
    print("=" * 70)
    print("GELU Derivative Polynomial Coefficient Generator")
    print("=" * 70)

    print(f"\nUsing {'mpmath (256-bit precision)' if HAVE_MPMATH else 'numpy (fp64)'} for reference")

    # First, understand the function shape
    print("\n" + "=" * 70)
    print("GELU'(x) Function Analysis")
    print("=" * 70)

    test_points = [-6, -4, -3, -2, -1, -0.85, -0.5, 0, 0.5, 1, 2, 3, 4, 6]
    print("\nKey values:")
    header = "GELU'(x)"
    print(f"  {'x':>8}  {header:>12}")
    print("  " + "-" * 22)
    for x in test_points:
        y = gelu_derivative(x)
        print(f"  {x:8.2f}  {y:12.6f}")

    # Find the minimum
    x_search = np.linspace(-2, 0, 1000)
    y_search = gelu_derivative(x_search)
    min_idx = np.argmin(y_search)
    min_x = x_search[min_idx]
    min_y = y_search[min_idx]
    print(f"\nLocal minimum: GELU'({min_x:.4f}) = {min_y:.6f}")

    # Try different polynomial degrees and ranges
    print("\n" + "=" * 70)
    print("Polynomial Fitting Experiments")
    print("=" * 70)

    # Experiment 1: Single polynomial over [-4, 4]
    print("\n--- Experiment 1: Single polynomial over [-4, 4] ---")
    for degree in [6, 8, 10, 12, 14]:
        print(f"\nDegree {degree}:")
        coeffs = fit_minimax_remez(gelu_derivative, degree, -4, 4)
        analyze_polynomial(coeffs, -4, 4, gelu_derivative)

    # Experiment 2: Single polynomial over [-5, 5]
    print("\n--- Experiment 2: Single polynomial over [-5, 5] ---")
    for degree in [8, 10, 12, 14]:
        print(f"\nDegree {degree}:")
        coeffs = fit_minimax_remez(gelu_derivative, degree, -5, 5)
        analyze_polynomial(coeffs, -5, 5, gelu_derivative)

    # Experiment 3: Single polynomial over [-3, 3] (active region only)
    print("\n--- Experiment 3: Single polynomial over [-3, 3] ---")
    for degree in [6, 8, 10, 12]:
        print(f"\nDegree {degree}:")
        coeffs = fit_minimax_remez(gelu_derivative, degree, -3, 3)
        analyze_polynomial(coeffs, -3, 3, gelu_derivative)

    # Find the best single polynomial
    print("\n" + "=" * 70)
    print("BEST SINGLE POLYNOMIAL")
    print("=" * 70)

    best_coeffs = None
    best_max_ulp = float("inf")
    best_config = None

    for x_range in [(-4, 4), (-5, 5), (-3, 3), (-4, 3), (-3, 4)]:
        for degree in [8, 10, 12, 14]:
            coeffs = fit_minimax_remez(gelu_derivative, degree, x_range[0], x_range[1])

            # Test over full BF16 range of interest
            x_test = np.linspace(-6, 6, 50000)
            y_exact = np.array([gelu_derivative(x) for x in x_test])
            y_poly = eval_poly(coeffs, x_test)

            # Apply clamping (like tanh does)
            y_poly = np.clip(y_poly, -0.18, 1.0)

            ulps = [compute_ulp_error(y_exact[i], y_poly[i]) for i in range(len(x_test))]
            max_ulp = max(ulps)

            if max_ulp < best_max_ulp:
                best_max_ulp = max_ulp
                best_coeffs = coeffs
                best_config = (x_range, degree)

    print(f"\nBest config: range={best_config[0]}, degree={best_config[1]}")
    print(f"Max ULP (with clamping): {best_max_ulp}")

    # Generate final coefficients
    print("\n" + "=" * 70)
    print("FINAL COEFFICIENTS")
    print("=" * 70)

    # Use degree 12 over [-4, 4] as a good balance
    final_range = (-4, 4)
    final_degree = 12

    print(f"\nGenerating degree-{final_degree} polynomial over {final_range}")
    final_coeffs = fit_minimax_remez(gelu_derivative, final_degree, final_range[0], final_range[1])

    print("\nCoefficients (for PolynomialEvaluator::eval):")
    print("// GELU'(x) polynomial approximation")
    print(f"// Degree {final_degree} over [{final_range[0]}, {final_range[1]}]")
    print("// Generated using Remez minimax algorithm")
    for i, c in enumerate(final_coeffs):
        print(f"//   c{i} = {c:+.18e}")

    # Detailed accuracy analysis
    print("\n" + "=" * 70)
    print("ACCURACY ANALYSIS (with clamping to [-0.18, 1.0])")
    print("=" * 70)

    # Test at different ranges
    ranges_to_test = [
        (-6, -4, "Deep negative (asymptotic)"),
        (-4, -2, "Negative transition"),
        (-2, -0.5, "Near minimum"),
        (-0.5, 0.5, "Near zero"),
        (0.5, 2, "Positive transition"),
        (2, 4, "Approaching 1"),
        (4, 6, "Large positive (asymptotic)"),
    ]

    print(f"\n{'Region':<30} {'Count':>6} {'Max ULP':>8} {'Mean ULP':>10} {'≤1 ULP':>8} {'≤2 ULP':>8}")
    print("-" * 80)

    total_ulps = []
    for x_min, x_max, name in ranges_to_test:
        x_test = np.linspace(x_min, x_max, 5000)
        y_exact = np.array([gelu_derivative(x) for x in x_test])
        y_poly = eval_poly(final_coeffs, x_test)
        y_poly = np.clip(y_poly, -0.18, 1.0)  # Apply clamping

        ulps = [compute_ulp_error(y_exact[i], y_poly[i]) for i in range(len(x_test))]
        total_ulps.extend(ulps)

        max_ulp = max(ulps)
        mean_ulp = np.mean(ulps)
        pct_le_1 = 100.0 * np.mean(np.array(ulps) <= 1)
        pct_le_2 = 100.0 * np.mean(np.array(ulps) <= 2)

        print(f"{name:<30} {len(ulps):>6} {max_ulp:>8} {mean_ulp:>10.2f} {pct_le_1:>7.1f}% {pct_le_2:>7.1f}%")

    print("-" * 80)
    total_ulps = np.array(total_ulps)
    print(
        f"{'OVERALL':<30} {len(total_ulps):>6} {np.max(total_ulps):>8} {np.mean(total_ulps):>10.2f} "
        f"{100.0*np.mean(total_ulps<=1):>7.1f}% {100.0*np.mean(total_ulps<=2):>7.1f}%"
    )

    # Generate C++ code
    print("\n" + "=" * 70)
    print("C++ IMPLEMENTATION")
    print("=" * 70)

    print(
        f"""
// ckernel_sfpu_gelu_derivative.h
// GELU'(x) polynomial approximation
// Degree {final_degree} over [{final_range[0]}, {final_range[1]}]
// With clamping to [-0.18, 1.0] for asymptotic regions

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_gelu_derivative_polynomial_(sfpi::vFloat x) {{
    // Polynomial coefficients (Remez minimax)
    sfpi::vFloat result = PolynomialEvaluator::eval(
        x,"""
    )

    for i, c in enumerate(final_coeffs):
        if i < len(final_coeffs) - 3:
            print(f"        {c:+.18e}f,  // c{i}")
        elif i == len(final_coeffs) - 3:
            print(f"        sfpi::vConstFloatPrgm2,  // c{i} = {c:+.18e}")
        elif i == len(final_coeffs) - 2:
            print(f"        sfpi::vConstFloatPrgm1,  // c{i} = {c:+.18e}")
        else:
            print(f"        sfpi::vConstFloatPrgm0); // c{i} = {c:+.18e}")

    print(
        f"""
    // Clamp to valid range [-0.18, 1.0]
    // GELU'(x) has minimum ≈ -0.17 at x ≈ -0.85
    // For x << 0: GELU'(x) → 0
    // For x >> 0: GELU'(x) → 1
    v_if(result < -0.18f) {{
        result = -0.18f;
    }}
    v_endif;
    v_if(result > 1.0f) {{
        result = sfpi::vConst1;
    }}
    v_endif;

    if constexpr (!is_fp32_acc_to_dest_mode) {{
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }}

    return result;
}}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void gelu_derivative_init() {{
    // Store last 3 polynomial coefficients in programmable registers
    sfpi::vConstFloatPrgm0 = {final_coeffs[-1]:+.18e}f;  // c{len(final_coeffs)-1}
    sfpi::vConstFloatPrgm1 = {final_coeffs[-2]:+.18e}f;  // c{len(final_coeffs)-2}
    sfpi::vConstFloatPrgm2 = {final_coeffs[-3]:+.18e}f;  // c{len(final_coeffs)-3}
}}
"""
    )

    # Test specific problematic points from bug report
    print("\n" + "=" * 70)
    print("BUG REPORT VALIDATION")
    print("=" * 70)
    print("\nTesting points from gelu_bw bug report:")

    bug_points = [-3.7, -3.719, -5.0, -10.0, 3.7, 5.0, 10.0]
    print(f"\n{'x':>8} {'Expected':>12} {'Poly':>12} {'Poly+Clamp':>12} {'ULP':>6}")
    print("-" * 56)

    for x in bug_points:
        expected = gelu_derivative(x)
        poly_raw = eval_poly(final_coeffs, np.array([x]))[0]
        poly_clamped = np.clip(poly_raw, -0.18, 1.0)
        ulp = compute_ulp_error(expected, poly_clamped)
        print(f"{x:8.3f} {expected:12.6e} {poly_raw:12.6e} {poly_clamped:12.6e} {ulp:6d}")


if __name__ == "__main__":
    main()
