#!/usr/bin/env python3
"""
Generate minimax polynomial coefficients for sech²(x) using scaled variables.

Two approaches compared:
  A) Map u = a² ∈ [0, 9) → t = (2/9)*u - 1 ∈ [-1, 1), fit p(t)
     Keeps even-symmetry benefit (half the coefficients vs direct-in-x).
     SFPU cost: 1 extra MAD to compute t from u.

  B) Map a ∈ [0, 3) → t = (2/3)*a - 1 ∈ [-1, 1), fit p(t)
     Needs ~2x the degree since we lose the u = a² trick.
     SFPU cost: 1 MAD to compute t from a (no a*a needed).

In both cases, |t| ≤ 1 so t^N stays bounded, avoiding the numerical
instability that makes degree-10+ blow up with the unscaled u ∈ [0, 9).
"""

import struct
import numpy as np
import mpmath

mpmath.mp.dps = 50


def sech2_mp(x):
    """High-precision sech²(x)."""
    return float(mpmath.sech(mpmath.mpf(str(x))) ** 2)


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
    sign = (bits >> 15) & 1
    mag = bits & 0x7FFF
    return -mag if sign else mag


def ulp_distance(bits_a, bits_b):
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


# =============================================================================
# Approach A: u = a², t = (2/9)*u - 1
# =============================================================================


def fit_poly_scaled_u(degree, num_samples=2000):
    """Fit polynomial in t where t = (2/9)*u - 1, u = x²."""
    # Chebyshev nodes on t ∈ [-1, 1]
    k = np.arange(num_samples)
    t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * num_samples))

    # Map back: u = (t + 1) * 9/2
    u_nodes = (t_nodes + 1.0) * 4.5

    # Target: sech²(√u)
    targets = np.array([sech2_mp(float(np.sqrt(u))) if u > 0 else 1.0 for u in u_nodes])
    weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

    V = np.vander(t_nodes, N=degree + 1, increasing=True)
    W = np.diag(weights)
    coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]
    return coeffs


def refine_scaled_u(coeffs, degree, iterations=8):
    """Iterative refinement for scaled-u approach."""
    num_base = 2000
    for _ in range(iterations):
        k = np.arange(num_base)
        t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * num_base))

        # Add error peak nodes
        bf16_values = get_all_bf16_in_range(0.0, 2.984375)
        error_t_nodes = []
        for bits, x in bf16_values:
            u = float(x * x)
            t = np.float32(u * (2.0 / 9.0) - 1.0)
            ref = sech2_mp(x)
            # Evaluate poly in t
            poly_val = sum(float(coeffs[i]) * float(t) ** i for i in range(len(coeffs)))
            ref_bf16 = float_to_bf16_rne(ref)
            poly_bf16 = float_to_bf16_rne(poly_val)
            ulp = ulp_distance(ref_bf16, poly_bf16)
            if ulp > 0:
                error_t_nodes.append(float(t))

        all_t = np.concatenate([t_nodes, np.array(error_t_nodes)])
        u_from_t = (all_t + 1.0) * 4.5
        targets = np.array([sech2_mp(float(np.sqrt(u))) if u > 0 else 1.0 for u in u_from_t])
        weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

        V = np.vander(all_t, N=degree + 1, increasing=True)
        W = np.diag(weights)
        coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]
    return coeffs


def validate_scaled_u(coeffs, verbose=True):
    """Validate scaled-u polynomial against all BF16 values in |x| < 3.0."""
    bf16_values = get_all_bf16_in_range(0.0, 2.984375)

    max_ulp = 0
    worst_x = 0
    total_ulp = 0
    ulp_histogram = {}

    for bits, x in bf16_values:
        ref = sech2_mp(x)
        ref_bf16_bits = float_to_bf16_rne(ref)

        # Simulate SFPU: a = |x|, u = a*a, t = u*(2/9) - 1, result = poly(t)
        a = np.float32(x)
        u = np.float32(a * a)
        t = np.float32(np.float32(u * np.float32(2.0 / 9.0)) + np.float32(-1.0))

        # Horner's in float32
        result = np.float32(0.0)
        for i in range(len(coeffs) - 1, -1, -1):
            result = np.float32(result * t + np.float32(coeffs[i]))

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


# =============================================================================
# Approach B: t = (2/3)*a - 1
# =============================================================================


def fit_poly_scaled_a(degree, num_samples=2000):
    """Fit polynomial in t where t = (2/3)*a - 1, a = |x|."""
    k = np.arange(num_samples)
    t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * num_samples))

    # Map back: a = (t + 1) * 3/2
    a_nodes = (t_nodes + 1.0) * 1.5

    targets = np.array([sech2_mp(float(a)) for a in a_nodes])
    weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

    V = np.vander(t_nodes, N=degree + 1, increasing=True)
    W = np.diag(weights)
    coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]
    return coeffs


def refine_scaled_a(coeffs, degree, iterations=8):
    """Iterative refinement for scaled-a approach."""
    num_base = 2000
    for _ in range(iterations):
        k = np.arange(num_base)
        t_nodes = np.cos(np.pi * (2 * k + 1) / (2 * num_base))

        bf16_values = get_all_bf16_in_range(0.0, 2.984375)
        error_t_nodes = []
        for bits, x in bf16_values:
            t = np.float32(np.float32(x * np.float32(2.0 / 3.0)) + np.float32(-1.0))
            ref = sech2_mp(x)
            poly_val = sum(float(coeffs[i]) * float(t) ** i for i in range(len(coeffs)))
            ref_bf16 = float_to_bf16_rne(ref)
            poly_bf16 = float_to_bf16_rne(poly_val)
            ulp = ulp_distance(ref_bf16, poly_bf16)
            if ulp > 0:
                error_t_nodes.append(float(t))

        all_t = np.concatenate([t_nodes, np.array(error_t_nodes)])
        a_from_t = (all_t + 1.0) * 1.5
        targets = np.array([sech2_mp(float(a)) for a in a_from_t])
        weights = 1.0 / np.maximum(np.abs(targets), 1e-10)

        V = np.vander(all_t, N=degree + 1, increasing=True)
        W = np.diag(weights)
        coeffs = np.linalg.lstsq(W @ V, W @ targets, rcond=None)[0]
    return coeffs


def validate_scaled_a(coeffs, verbose=True):
    """Validate scaled-a polynomial against all BF16 values in |x| < 3.0."""
    bf16_values = get_all_bf16_in_range(0.0, 2.984375)

    max_ulp = 0
    worst_x = 0
    total_ulp = 0
    ulp_histogram = {}

    for bits, x in bf16_values:
        ref = sech2_mp(x)
        ref_bf16_bits = float_to_bf16_rne(ref)

        # Simulate SFPU: a = |x|, t = a*(2/3) - 1, result = poly(t)
        a = np.float32(x)
        t = np.float32(np.float32(a * np.float32(2.0 / 3.0)) + np.float32(-1.0))

        result = np.float32(0.0)
        for i in range(len(coeffs) - 1, -1, -1):
            result = np.float32(result * t + np.float32(coeffs[i]))

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


def main():
    print("=" * 70)
    print("APPROACH A: Scaled u = a², t = (2/9)*u - 1 ∈ [-1, 1)")
    print("  (keeps even-symmetry benefit, same coefficient count as unscaled)")
    print("=" * 70)

    for degree in range(8, 14):
        print(f"\n--- Degree {degree} in t (= degree {2*degree} in x) ---")
        coeffs = fit_poly_scaled_u(degree)
        coeffs = refine_scaled_u(coeffs, degree, iterations=8)
        print("Validation:")
        max_ulp, worst_x, mean_ulp = validate_scaled_u(coeffs, verbose=True)

        if max_ulp <= 1:
            print(f"\n  *** Degree {degree} achieves Max ULP = {max_ulp}! ***")
            print(f"\nC++ constexpr declarations:")
            print(f"// Approach A: t = u * (2.0f/9.0f) - 1.0f, where u = a * a")
            for i, c in enumerate(coeffs):
                print(f"constexpr float SECH2_SCALED_C{i} = {c:.20e}f;")
            args = ", ".join([f"SECH2_SCALED_C{i}" for i in range(len(coeffs))])
            print(f"\nresult = PolynomialEvaluator::eval(t, {args});")

    print(f"\n\n{'=' * 70}")
    print("APPROACH B: Scaled a = |x|, t = (2/3)*a - 1 ∈ [-1, 1)")
    print("  (no a² needed, but needs ~2x degree for same accuracy)")
    print("=" * 70)

    for degree in range(14, 22):
        print(f"\n--- Degree {degree} in t (= degree {degree} in x) ---")
        coeffs = fit_poly_scaled_a(degree)
        coeffs = refine_scaled_a(coeffs, degree, iterations=8)
        print("Validation:")
        max_ulp, worst_x, mean_ulp = validate_scaled_a(coeffs, verbose=True)

        if max_ulp <= 1:
            print(f"\n  *** Degree {degree} achieves Max ULP = {max_ulp}! ***")
            print(f"\nC++ constexpr declarations:")
            print(f"// Approach B: t = a * (2.0f/3.0f) - 1.0f, where a = abs(x)")
            for i, c in enumerate(coeffs):
                print(f"constexpr float SECH2_SCALED_C{i} = {c:.20e}f;")
            args = ", ".join([f"SECH2_SCALED_C{i}" for i in range(len(coeffs))])
            print(f"\nresult = PolynomialEvaluator::eval(t, {args});")
            break


if __name__ == "__main__":
    main()
