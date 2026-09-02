// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu
{

/**
 * @brief Compile-time polynomial evaluator.
 *
 * This template struct provides efficient polynomial evaluation at both compile-time
 * and runtime. The polynomial is represented by coefficients in ascending order of powers:
 * coef[0] + coef[1]*x + coef[2]*x^2 + ... + coef[N-1]*x^(N-1)
 *
 * Two evaluation schemes are used:
 *  - Horner's method (a single serial multiply-add chain) for low-degree polynomials.
 *  - A second-order (even/odd) Horner split for high-degree polynomials. The polynomial is
 *    factored as P(x) = E(x^2) + x * O(x^2), where E collects the even-power coefficients and
 *    O the odd-power ones. E(x^2) and O(x^2) are independent Horner chains, so their multiply-adds
 *    can be interleaved/pipelined by the compiler, roughly halving the critical-path latency at the
 *    cost of one extra multiply (x^2), one final combine, and one extra live register.
 *
 * The split kicks in only once the number of coefficients reaches SplitThreshold, because for short
 * chains the extra x^2 / combine overhead outweighs the shorter dependency chain.
 *
 * Kept in sync with the Blackhole evaluator (tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_polyval.h)
 * so kernels ported between the two architectures evaluate their polynomials identically.
 *
 * @note Horner's method is used for numerical stability and O(N) complexity.
 * @note For N == 0, returns U{0}. For N == 1, returns the constant term.
 * @note Switching schemes changes floating-point rounding, so results for a polynomial evaluated with
 *       the split differ slightly (in the last bits) from the pure-Horner result. Passing a
 *       SplitThreshold above the coefficient count pins a call site to plain Horner, which
 *       register-tight kernels need: the split keeps ~2 extra LRegs live.
 *
 * @see https://en.wikipedia.org/wiki/Horner%27s_method
 * @see https://en.wikipedia.org/wiki/Estrin%27s_scheme
 */
struct PolynomialEvaluator
{
    // Number of coefficients at (or above) which the even/odd interleaved scheme is used instead of
    // plain Horner. 6 is the crossover where the split's critical path (ceil(N/2)-1 chain + x^2 + combine)
    // first becomes strictly shorter than Horner's N-1 serial multiply-adds.
    static constexpr int DefaultSplitThreshold = 6;

private:
    // Holds the two independent partial results of the even/odd split. The even and odd accumulators
    // may have different types (e.g. one purely scalar, one sfpi::vFloat), hence two type parameters.
    template <typename E, typename O>
    struct EvenOdd {
        E even;
        O odd;
    };

    // Plain Horner's method: coeff0 + x * (coeff1 + x * (...)). Serial multiply-add chain.
    template <typename U>
    sfpi_inline static constexpr auto horner([[maybe_unused]] U x) {
        return U {0};
    }

    template <typename U, typename Coefficient0>
    sfpi_inline static constexpr auto horner([[maybe_unused]] U x, Coefficient0 coeff0) {
        return coeff0;
    }

    template <typename U, typename Coefficient0, typename... OtherCoefficients>
    sfpi_inline static constexpr auto horner(U x, Coefficient0 coeff0, OtherCoefficients... other_coefficients) {
        return coeff0 + x * horner(x, other_coefficients...);
    }

    // Even/odd split evaluated in y = x^2. Peels two coefficients per step (one even, one odd) and
    // threads two independent Horner accumulators so the resulting multiply-adds can be interleaved.
    template <typename U>
    sfpi_inline static constexpr auto split([[maybe_unused]] U y) {
        return EvenOdd<U, U>{U{0}, U{0}};
    }

    template <typename U, typename Coefficient0>
    sfpi_inline static constexpr auto split([[maybe_unused]] U y, Coefficient0 coeff0) {
        return EvenOdd<Coefficient0, U>{coeff0, U{0}};
    }

    template <typename U, typename Coefficient0, typename Coefficient1>
    sfpi_inline static constexpr auto split([[maybe_unused]] U y, Coefficient0 coeff0, Coefficient1 coeff1) {
        return EvenOdd<Coefficient0, Coefficient1>{coeff0, coeff1};
    }

    template <typename U, typename Coefficient0, typename Coefficient1, typename... OtherCoefficients>
    sfpi_inline static constexpr auto split(
        U y, Coefficient0 coeff0, Coefficient1 coeff1, OtherCoefficients... other_coefficients) {
        auto rest = split(y, other_coefficients...);
        auto even = coeff0 + y * rest.even;
        auto odd = coeff1 + y * rest.odd;
        return EvenOdd<decltype(even), decltype(odd)>{even, odd};
    }

public:
    /**
     * @brief Evaluates the polynomial at the given point.
     *
     * @tparam SplitThreshold Minimum number of coefficients required to switch from plain Horner to the
     *         interleaved even/odd scheme. Defaults to DefaultSplitThreshold.
     * @param x The point at which to evaluate the polynomial
     * @param coefficients Polynomial coefficients in ascending order of powers.
     * @return The value of the polynomial at the given point
     *
     * @note Coefficients can be either float, sfpi::vFloat, ... (scalar and sfpi typed arguments can be mixed)
     */
    template <int SplitThreshold = DefaultSplitThreshold, typename U, typename... Coefficients>
    sfpi_inline static constexpr auto eval(U x, Coefficients... coefficients) {
        if constexpr (static_cast<int>(sizeof...(Coefficients)) >= SplitThreshold) {
            // P(x) = E(x^2) + x * O(x^2), with E and O evaluated as independent (interleavable) chains.
            U x2 = x * x;
            auto parts = split(x2, coefficients...);
            return parts.even + x * parts.odd;
        } else {
            return horner(x, coefficients...);
        }
    }
};

} // namespace ckernel::sfpu
