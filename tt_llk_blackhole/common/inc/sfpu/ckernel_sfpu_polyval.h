// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

template <typename T>
constexpr T POLYVAL5(T coef4, T coef3, T coef2, T coef1, T coef0, T val)
{
    return (((((coef4 * val) + coef3) * val + coef2) * val + coef1) * val + coef0);
}

template <typename T>
constexpr T POLYVAL7(T coef6, T coef5, T coef4, T coef3, T coef2, T coef1, T coef0, T val)
{
    return (((((((coef6 * val) + coef5) * val + coef4) * val + coef3) * val + coef2) * val + coef1) * val + coef0);
}

/**
 * @brief Compile-time polynomial evaluator using Horner's method.
 *
 * This template struct provides efficient polynomial evaluation at both compile-time
 * and runtime using Horner's method (synthetic division). The implementation uses
 * a simple loop instead of template recursion for better compiler optimization.
 *
 * The polynomial is represented by coefficients in ascending order of powers:
 * coef[0] + coef[1]*x + coef[2]*x^2 + ... + coef[N-1]*x^(N-1)
 *
 * @tparam N The degree of the polynomial plus one (number of coefficients)
 * @tparam T The numeric type for evaluation point (e.g., sfpi::vFloat)
 * @tparam U The numeric type for coefficients (e.g., float, double, int)
 *
 * @note This implementation is constexpr-compatible and can be evaluated at compile time.
         Make sure to mark coefficients array as constexpr to leverage compile-time evaluation,
         otherwise evaluation will be performed at runtime.
 * @note Uses Horner's method for numerical stability and O(N) complexity.
 * @note For N <= 0, returns T{0}. For N == 1, returns the constant term.
 *
 * @see https://en.wikipedia.org/wiki/Horner%27s_method
 */
template <int N, typename T, typename U>
struct PolynomialEvaluator
{
    /**
     * @brief Evaluates the polynomial at the given point.
     *
     * @param coefficients Pointer to array of N coefficients in ascending order of powers
     * @param val The point at which to evaluate the polynomial
     * @return The value of the polynomial at the given point
     *
     * @pre coef must point to at least N valid coefficients
     * @pre N should be positive for meaningful results
     *
     */
    static constexpr T eval(const U* coefficients, T val)
    {
        T result = (N <= 0) ? T {0} : T {coefficients[N - 1]};
        for (int i = N - 2; i >= 0; --i)
        {
            result = result * val + T {coefficients[i]};
        }
        return result;
    }
};

} // namespace ckernel::sfpu
