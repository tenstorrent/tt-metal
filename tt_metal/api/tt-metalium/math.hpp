// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
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
 * @return The result of ceiling division (a + b - 1) / b.
 *
 * @note If b is zero, this results in undefined behavior.
 */
template <typename A, typename B>
constexpr auto div_up(A a, B b) noexcept -> std::common_type_t<A, B> {
    using T = std::common_type_t<A, B>;
    assert(b != 0 && "Divide by zero error in div_up");
    return static_cast<T>((static_cast<T>(a) + static_cast<T>(b) - 1) / static_cast<T>(b));
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
