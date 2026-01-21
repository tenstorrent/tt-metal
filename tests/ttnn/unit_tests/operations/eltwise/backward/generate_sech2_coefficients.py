# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generate polynomial coefficients for sech²(x) = 1/cosh²(x) approximation.
Uses Remez exchange algorithm for minimax polynomial approximation.
"""

import numpy as np
import struct


def sech2(x):
    """Compute sech²(x) = 1/cosh²(x) with numerical stability."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    result = np.zeros_like(x)
    large = x > 20
    small = ~large
    result[large] = 4.0 * np.exp(-2 * x[large])
    result[small] = 1.0 / np.cosh(x[small]) ** 2
    return result


def float_to_bf16(f):
    """Convert float32 to bfloat16."""
    bits = struct.unpack(">I", struct.pack(">f", float(f)))[0]
    return bits >> 16


def bf16_to_float(bf16_bits):
    """Convert bfloat16 to float32."""
    bits = int(bf16_bits) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def eval_poly(coeffs, x):
    """Evaluate polynomial using Horner's method."""
    result = np.zeros_like(np.asarray(x, dtype=np.float64))
    for c in reversed(coeffs):
        result = result * x + c
    return result


def fit_minimax_remez(func, degree, x_min, x_max, max_iters=100, tol=1e-12):
    """Remez exchange algorithm for minimax polynomial approximation."""
    n = degree + 1

    # Initial reference points (Chebyshev nodes)
    k = np.arange(n + 1)
    ref_points = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * np.cos(np.pi * k / n)
    ref_points = np.sort(ref_points)

    coeffs = None

    for _ in range(max_iters):
        A = np.zeros((n + 1, n + 1))
        b = func(ref_points)

        for i, x in enumerate(ref_points):
            for j in range(n):
                A[i, j] = x**j
            A[i, n] = (-1) ** i

        try:
            sol = np.linalg.solve(A, b)
            coeffs = sol[:n]
        except np.linalg.LinAlgError:
            ref_points += np.random.randn(n + 1) * 1e-8
            continue

        x_dense = np.linspace(x_min, x_max, 20000)
        errors = func(x_dense) - eval_poly(coeffs, x_dense)

        extrema_indices = [0]
        for i in range(1, len(errors) - 1):
            if (errors[i - 1] < errors[i] > errors[i + 1]) or (errors[i - 1] > errors[i] < errors[i + 1]):
                extrema_indices.append(i)
        extrema_indices.append(len(errors) - 1)

        extrema_x = x_dense[extrema_indices]
        extrema_err = np.abs(errors[extrema_indices])

        sorted_idx = np.argsort(-extrema_err)
        new_ref = np.sort(extrema_x[sorted_idx[: n + 1]])

        if np.max(np.abs(new_ref - ref_points)) < tol:
            break
        ref_points = new_ref

    return coeffs


def compute_ulp_error(expected, actual):
    """Compute ULP distance between two BF16 values."""
    if expected == 0 and actual == 0:
        return 0

    exp_bf16 = float_to_bf16(expected)
    act_bf16 = float_to_bf16(actual)

    if exp_bf16 == act_bf16:
        return 0

    def bf16_to_signed_int(bf16):
        if bf16 >= 0x8000:
            return -(0x10000 - bf16)
        return bf16

    return abs(bf16_to_signed_int(exp_bf16) - bf16_to_signed_int(act_bf16))


def main():
    print("=" * 70)
    print("sech²(x) Polynomial Coefficient Generator")
    print("=" * 70)

    # Generate best polynomial for implementation
    # Using [0, 4.0] range with degree 12 for good coverage
    x_max = 4.0
    degree = 12

    print(f"\nGenerating Remez minimax polynomial (degree {degree}) over [0, {x_max}]")
    coeffs = fit_minimax_remez(sech2, degree, 0, x_max)

    print("\nCoefficients:")
    for i, c in enumerate(coeffs):
        print(f"  c{i} = {c:+.18e}")

    # Analyze accuracy
    x_test = np.linspace(0, 5, 50000)
    y_exact = sech2(x_test)
    y_poly = eval_poly(coeffs, x_test)
    y_poly = np.clip(y_poly, 0, 1)  # Clamp

    print("\nAccuracy Analysis:")
    for x_max_test in [3.5, 4.0, 4.5, 5.0]:
        mask = x_test <= x_max_test
        ulps = []
        for i in np.where(mask)[0]:
            ulps.append(compute_ulp_error(y_exact[i], y_poly[i]))
        ulps = np.array(ulps)
        print(
            f"  [0, {x_max_test}]: Max ULP = {np.max(ulps)}, Mean = {np.mean(ulps):.2f}, "
            f"<= 2 ULP: {100*np.mean(ulps <= 2):.1f}%"
        )

    # Test specific points
    print("\nKey test points:")
    test_points = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.3438, 3.5, 4, 4.5, 5]
    for x in test_points:
        exact = sech2(x)
        poly = max(0, min(1, eval_poly(coeffs, x)))
        ulp = compute_ulp_error(exact, poly)
        status = "OK" if ulp <= 2 else ("WARN" if ulp <= 10 else "BAD")
        print(f"  x={x:5.2f}: exact={exact:.6e}, poly={poly:.6e}, ULP={ulp:5d} [{status}]")

    # Generate C++ code
    print("\n" + "=" * 70)
    print("C++ IMPLEMENTATION")
    print("=" * 70)

    # Store some coefficients in programmable registers (like tanh does)
    # vConstFloatPrgm0, vConstFloatPrgm1, vConstFloatPrgm2
    print(
        f"""
// sech²(x) = 1/cosh²(x) polynomial approximation
// Remez minimax over [0, {x_max}] with degree {degree}
// Avoids precision loss from 1-tanh²(x) computation

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sech2_polynomial_(sfpi::vFloat x) {{
    sfpi::vFloat val = sfpi::abs(x);  // sech²(-x) = sech²(x)

    // Polynomial coefficients (Remez minimax)
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,"""
    )

    for i, c in enumerate(coeffs):
        suffix = "," if i < len(coeffs) - 1 else ");"
        # Use programmable constants for last 3 coefficients
        if i >= len(coeffs) - 3:
            print(f"        sfpi::vConstFloatPrgm{len(coeffs) - 1 - i}{suffix}  // c{i} = {c:.18e}")
        else:
            print(f"        {c:.18e}f{suffix}  // c{i}")

    print(
        """
    // Clamp to [0, 1]
    sfpi::vFloat one = sfpi::vConst1;
    sfpi::vFloat zero = sfpi::vConst0;
    sfpi::vec_min_max(result, one);
    sfpi::vec_min_max(zero, result);

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void sech2_init() {
    // Store last 3 polynomial coefficients in programmable registers"""
    )

    for i in range(3):
        idx = len(coeffs) - 1 - i
        print(f"    sfpi::vConstFloatPrgm{i} = {coeffs[idx]:.18e}f;")

    print("}")

    # Comparison with original bug
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER ORIGINAL BUG")
    print("=" * 70)
    print(
        """
Original bug (1 - tanh²):
  x=3.3438: expected=0.0049, got=0.0000, ULP=15,139
  x=4.0000: expected=0.0013, got=0.0000, ULP=14,896
  x=5.0000: expected=0.0002, got=0.0000, ULP=14,515

With polynomial fix:"""
    )

    for x in [3.3438, 4.0, 5.0]:
        exact = sech2(x)
        poly = max(0, min(1, eval_poly(coeffs, x)))
        ulp = compute_ulp_error(exact, poly)
        print(f"  x={x}: expected={exact:.4e}, got={poly:.4e}, ULP={ulp}")


if __name__ == "__main__":
    main()
