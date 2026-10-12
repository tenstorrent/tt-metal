// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <type_traits>
#include <utility>

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
 * @brief Rounds up a to a multiple of b.
 *
 * Computes the smallest multiple of b that is greater than or equal to a. The sign of b is ignored
 * (the multiples of b and -b are the same), so the result is always >= a.
 *
 * @param a The number to round.
 * @param b The multiple to round up to. Must be non-zero.
 * @return The rounded-up value.
 *
 * @note b must be non-zero (checked with assert). The mathematically rounded result must be representable in
 * the result type; otherwise it wraps (unsigned) or is undefined (signed). For signed types this includes
 * b == min() (|b| is not representable) and a == min() with b == -1.
 */
template <typename A, typename B>
constexpr auto round_up(A a, B b) noexcept {
    using T = decltype(+std::declval<std::common_type_t<A, B>>());
    const T value = static_cast<T>(a);
    const T multiple = static_cast<T>(b);
    assert(multiple != 0 && "Divide by zero error in round_up");
    const T remainder = value % multiple;
    if constexpr (std::is_signed_v<T>) {
        const T abs_multiple = multiple < 0 ? static_cast<T>(-multiple) : multiple;
        return static_cast<T>((value - remainder) + (remainder > 0 ? abs_multiple : static_cast<T>(0)));
    } else {
        return static_cast<T>((value - remainder) + (remainder != 0 ? multiple : static_cast<T>(0)));
    }
}

/**
 * @brief Rounds down a to a multiple of b.
 *
 * Computes the largest multiple of b that is less than or equal to a. The sign of b is ignored
 * (the multiples of b and -b are the same), so the result is always <= a.
 *
 * @param a The number to round.
 * @param b The multiple to round down to. Must be non-zero.
 * @return The rounded-down value.
 *
 * @note b must be non-zero (checked with assert). The mathematically rounded result must be representable in
 * the result type; otherwise it wraps (unsigned) or is undefined (signed). For signed types this includes
 * b == min() (|b| is not representable) and a == min() with b == -1.
 */
template <typename A, typename B>
constexpr auto round_down(A a, B b) noexcept {
    using T = decltype(+std::declval<std::common_type_t<A, B>>());
    const T value = static_cast<T>(a);
    const T multiple = static_cast<T>(b);
    assert(multiple != 0 && "Divide by zero error in round_down");
    const T remainder = value % multiple;
    if constexpr (std::is_signed_v<T>) {
        const T abs_multiple = multiple < 0 ? static_cast<T>(-multiple) : multiple;
        return static_cast<T>((value - remainder) - (remainder < 0 ? abs_multiple : static_cast<T>(0)));
    } else {
        return static_cast<T>(value - remainder);
    }
}

}  // namespace tt
