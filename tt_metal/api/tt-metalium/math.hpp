// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <type_traits>

namespace tt {

/**
 * @brief Computes the ceiling of a / b.
 *
 * Returns the smallest integer greater than or equal to a / b.
 *
 * @param a The numerator.
 * @param b The denominator. Must be non-zero.
 * @return The result of ceiling division.
 *
 * @note If b is zero, this results in undefined behavior.
 */
template <typename A, typename B>
constexpr auto div_up(A a, B b) noexcept -> std::common_type_t<A, B> {
    using T = std::common_type_t<A, B>;
    const T numerator = static_cast<T>(a);
    const T denominator = static_cast<T>(b);
    assert(denominator != 0 && "Divide by zero error in div_up");
    const T quotient = numerator / denominator;
    const T remainder = numerator % denominator;
    if constexpr (std::is_signed_v<T>) {
        return quotient + static_cast<T>(remainder != 0 && (remainder > 0) == (denominator > 0));
    }
    return quotient + static_cast<T>(remainder != 0);
}

/**
 * @brief Rounds up a to the nearest multiple of b.
 *
 * Computes the smallest multiple of b that is greater than or equal to a.
 *
 * @param a The number to round.
 * @param b The multiple to round up to. Must be non-zero.
 * @return The rounded-up value.
 *
 * @note Internally uses div_up. If b is zero, this results in undefined behavior.
 */
template <typename A, typename B>
constexpr auto round_up(A a, B b) {
    using T = std::common_type_t<A, B>;
    return static_cast<T>(b) * div_up(static_cast<T>(a), static_cast<T>(b));
}
/**
 * @brief Rounds down a to the nearest multiple of b.
 *
 * Computes the largest multiple of b that is less than or equal to a.
 *
 * @param a The number to round.
 * @param b The multiple to round down to. Must be non-zero.
 * @return The rounded-down value.
 *
 * @note If b is zero, this results in undefined behavior.
 */
template <typename A, typename B>
constexpr auto round_down(A a, B b) {
    using T = std::common_type_t<A, B>;
    return static_cast<T>(b) * (static_cast<T>(a) / static_cast<T>(b));
}

/**
 * @brief Rounds addr up to the nearest multiple of alignment.
 *
 * Use this function to pad a buffer size or to move an address to an alignment boundary.
 * The value of addr does not have to be a power of two.
 * If addr is already a multiple of alignment, the function returns addr.
 *
 * Examples:
 * - align(17u, 16u) returns 32.
 * - align(32u, 16u) returns 32.
 * - align(0u, 16u) returns 0.
 *
 * @tparam T1 Integral type of the address or size.
 * @tparam T2 Integral type of the alignment.
 * @param addr Address, offset, or size to round up. This value must not be negative.
 * @param alignment Alignment boundary, in the same units as addr (for example, bytes).
 *        This value must be a positive power of two.
 * @return The smallest multiple of alignment that is equal to or greater than addr.
 *         The return type is std::common_type_t<T1, T2>.
 * @pre alignment must be a positive power of two.
 *      If NDEBUG is defined, this check is omitted; callers must still satisfy the precondition.
 * @pre The result must fit in the return type. The function does not check for overflow.
 * @note You can use this function in constant expressions.
 */
template <typename T1, typename T2>
constexpr std::common_type_t<T1, T2> align(T1 addr, T2 alignment) {
    static_assert(std::is_integral_v<T1>, "align() requires integral types");
    static_assert(std::is_integral_v<T2>, "align() requires integral types");
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0 && "align() requires a positive power of two alignment");
    using T = std::common_type_t<T1, T2>;
    // This bit-mask calculation is correct only when alignment is a power of two.
    return ((static_cast<T>(addr) - 1) | (static_cast<T>(alignment) - 1)) + 1;
}

}  // namespace tt
