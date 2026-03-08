#!/usr/bin/env python3
"""
Generate minimax polynomial coefficients for sech²(x) as p(u) where u = x².

Since sech²(x) is even, sech²(x) = f(x²) = f(u).
We fit p(u) = c0 + c1*u + c2*u² + ... + cN*u^N over u ∈ [0, 9] (i.e., |x| < 3).
This gives degree-2N accuracy in x with only N+1 coefficients.

The polynomial is evaluated via Horner's method using PolynomialEvaluator::eval(u, c0, c1, ..., cN).
"""

import struct
import numpy as np
import mpmath

mpmath.mp.dps = 50


def sech2_mp(x):
    """High-precision sech²(x)."""
    return float(mpmath.sech(mpmath.mpf(str(x))) ** 2)


def sech2_of_sqrt_u_mp(u):
    """High-precision f(u) = sech²(√u)."""
    if u <= 0:
        return 1.0
    return sech2_mp(float(mpmath.sqrt(mpmath.mpf(str(u)))))


def float_to_bf16_rne(f):
    """Convert float to BF16 bits using RNE."""
    packed = struct.pack(">f", f)
    bits32 = struct.unpack(">I", packed)[0]
    sign = (bits32 >> 31) & 1
    exp = (bits32 >> 23) & 0xFF
    mantissa = bits32 & 0x7FFFFF
    if exp == 0xFF:
        return (sign << 15) | (0xFF << 7) | (mantissa >> 16)
    if exp == 0:
        return sign << 15
    round_bit = (mantissa >> 15) & 1
    sticky = (mantissa & 0x7FFF) != 0
    truncated = mantissa >> 16
    if round_bit and (sticky or (truncated & 1)):
        truncated += 1
        if truncated > 0x7F:
            truncated = 0
            exp += 1
            if exp >= 0xFF:
                return (sign << 15) | (0xFF << 7)
    return (sign << 15) | (exp << 7) | truncated


def bf16_bits_to_float(bits):
    """Convert BF16 bits to float."""
    bits32 = bits << 16
    packed = struct.pack(">I", bits32)
    return struct.unpack(">f", packed)[0]


def bf16_order_index(bits):
    """BF16 order index for ULP distance."""
    sign = (bits >> 15) & 1
    mag = bits & 0x7FFF
    return -mag if sign else mag


def ulp_distance(bits_a, bits_b):
    """ULP distance between two BF16 values."""
    return abs(bf16_order_index(bits_a) - bf16_order_index(bits_b))


def get_all_bf16_in_range(x_min, x_max):
    """Get all positive normal BF16 values in [x_min, x_max]."""
    values = []
    for exp in range(1, 255):
        for mantissa in range(128):
            bits = (exp << 7) | mantissa
            f = bf16_bits_to_float(bits)
            if x_min <= f <= x_max:
                values.append((bits, f))
    return values


def fit_polynomial(degree, num_samples=2000):
    """Fit minimax-style polynomial using weighted Chebyshev nodes."""
    u_min, u_max = 0.0, 9.0

    # Chebyshev nodes on [u_min, u_max]
    k = np.arange(num_samples)
    nodes = 0.5 * (u_min + u_max) + 0.5 * (u_max - u_min) * np.cos(np.pi * (2 * k + 1) / (2 * num_samples))

    # Evaluate target function at nodes
    targets = np.array([sech2_of_sqrt_u_mp(u) for u in nodes])

    # Weight by 1/|f(u)| to get relative error minimization
    # (clamped to avoid division by very small values)
    weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

    # Weighted least squares
    V = np.vander(nodes, N=degree + 1, increasing=True)
    W = np.diag(weights)
    coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]

    return coeffs


def validate_polynomial(coeffs, verbose=True):
    """Validate polynomial against all BF16 values in |x| < 3.0."""
    bf16_values = get_all_bf16_in_range(0.0, 2.984375)  # largest BF16 < 3.0

    max_ulp = 0
    worst_x = 0
    total_ulp = 0
    ulp_histogram = {}

    for bits, x in bf16_values:
        # Reference
        ref = sech2_mp(x)
        ref_bf16_bits = float_to_bf16_rne(ref)

        # Polynomial evaluation in FP32 (simulates hardware)
        u = np.float32(x) * np.float32(x)
        # Horner's method in float32
        result = np.float32(0.0)
        for i in range(len(coeffs) - 1, -1, -1):
            result = np.float32(result * u + np.float32(coeffs[i]))

        result_bf16_bits = float_to_bf16_rne(float(result))
        ulp = ulp_distance(result_bf16_bits, ref_bf16_bits)

        if ulp > max_ulp:
            max_ulp = ulp
            worst_x = x
        total_ulp += ulp
        ulp_histogram[ulp] = ulp_histogram.get(ulp, 0) + 1

    mean_ulp = total_ulp / len(bf16_values) if bf16_values else 0

    if verbose:
        print(f"  Values tested: {len(bf16_values)}")
        print(f"  Max ULP: {max_ulp} (at x = {worst_x})")
        print(f"  Mean ULP: {mean_ulp:.4f}")
        for ulp_val in sorted(ulp_histogram.keys()):
            count = ulp_histogram[ulp_val]
            pct = 100.0 * count / len(bf16_values)
            print(f"    ULP = {ulp_val}: {count} values ({pct:.2f}%)")

    return max_ulp, worst_x, mean_ulp


def validate_with_negative(coeffs, verbose=True):
    """Validate for both positive and negative x (sech² is even)."""
    bf16_values = get_all_bf16_in_range(0.0, 2.984375)

    max_ulp = 0
    worst_x = 0
    total = 0
    count = 0

    for bits, x in bf16_values:
        for sign in [1.0, -1.0]:
            xv = sign * x
            ref = sech2_mp(xv)
            ref_bf16_bits = float_to_bf16_rne(ref)

            # Simulate: a = |x|, u = a*a, result = poly(u)
            a = np.float32(abs(xv))
            u = np.float32(a * a)
            result = np.float32(0.0)
            for i in range(len(coeffs) - 1, -1, -1):
                result = np.float32(result * u + np.float32(coeffs[i]))

            result_bf16_bits = float_to_bf16_rne(float(result))
            ulp = ulp_distance(result_bf16_bits, ref_bf16_bits)

            if ulp > max_ulp:
                max_ulp = ulp
                worst_x = xv
            total += ulp
            count += 1

    mean_ulp = total / count if count else 0
    if verbose:
        print(f"  Values tested (pos+neg): {count}")
        print(f"  Max ULP: {max_ulp} (at x = {worst_x})")
        print(f"  Mean ULP: {mean_ulp:.4f}")

    return max_ulp


def iterative_remez_refinement(coeffs, degree, iterations=5):
    """Iteratively refine polynomial by adding error peaks as extra nodes."""
    u_min, u_max = 0.0, 9.0
    num_base = 2000

    for iteration in range(iterations):
        # Base Chebyshev nodes
        k = np.arange(num_base)
        nodes = 0.5 * (u_min + u_max) + 0.5 * (u_max - u_min) * np.cos(np.pi * (2 * k + 1) / (2 * num_base))

        # Add error peak nodes from BF16 validation
        bf16_values = get_all_bf16_in_range(0.0, 2.984375)
        error_nodes = []
        for bits, x in bf16_values:
            u = float(x * x)
            ref = sech2_mp(x)
            poly_val = sum(float(coeffs[i]) * u**i for i in range(len(coeffs)))
            ref_bf16 = float_to_bf16_rne(ref)
            poly_bf16 = float_to_bf16_rne(poly_val)
            ulp = ulp_distance(ref_bf16, poly_bf16)
            if ulp > 0:
                error_nodes.append(u)

        all_nodes = np.concatenate([nodes, np.array(error_nodes)])
        targets = np.array([sech2_of_sqrt_u_mp(u) for u in all_nodes])
        weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

        V = np.vander(all_nodes, N=degree + 1, increasing=True)
        W = np.diag(weights)
        coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]

    return coeffs


def main():
    print("=" * 70)
    print("Polynomial Coefficient Generation for sech²(x) = p(u), u = x²")
    print("Range: u ∈ [0, 9] (|x| < 3.0)")
    print("=" * 70)

    best_coeffs = None
    best_ulp = 999
    best_degree = 0

    for degree in range(6, 12):
        print(f"\n--- Degree {degree} in u (= degree {2*degree} in x) ---")

        # Initial fit
        coeffs = fit_polynomial(degree)

        # Iterative refinement
        coeffs = iterative_remez_refinement(coeffs, degree, iterations=8)

        print(f"Validation (positive x only):")
        max_ulp, worst_x, mean_ulp = validate_polynomial(coeffs, verbose=True)

        if max_ulp < best_ulp:
            best_ulp = max_ulp
            best_coeffs = coeffs
            best_degree = degree

        if max_ulp <= 1:
            print(f"\n  *** Degree {degree} achieves Max ULP = {max_ulp}! ***")
            # Also validate with negatives
            print(f"Validation (positive + negative x):")
            validate_with_negative(coeffs, verbose=True)
            break

    print(f"\n{'=' * 70}")
    print(f"BEST RESULT: Degree {best_degree}, Max ULP = {best_ulp}")
    print(f"{'=' * 70}")

    if best_coeffs is not None:
        print(f"\nC++ constexpr declarations:")
        print()
        for i, c in enumerate(best_coeffs):
            print(f"constexpr float SECH2_POLY_C{i} = {c:.20e}f;")

        print(f"\nPolynomialEvaluator::eval call:")
        args = ", ".join([f"SECH2_POLY_C{i}" for i in range(len(best_coeffs))])
        print(f"result = PolynomialEvaluator::eval(u, {args});")

        # Final comprehensive validation
        print(f"\n--- Final validation with both signs ---")
        validate_with_negative(best_coeffs, verbose=True)


if __name__ == "__main__":
    main()
