// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>
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

}  // namespace tt
