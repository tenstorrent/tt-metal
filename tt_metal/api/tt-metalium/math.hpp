// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
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

/**
 * @brief Converts a floating-point value to an integral type, clamping instead of overflowing.
 *
 * A plain static_cast from a floating type to an integral type has undefined behavior whenever the
 * truncated value cannot be represented in the destination, and the host architectures do not agree
 * on what it produces. x86_64 emits cvtt* and yields the "integer indefinite" pattern, so 1e30f
 * becomes INT32_MIN and -1.0f becomes 0xFFFFFFFF as a uint32_t; aarch64 emits fcvtz* and saturates,
 * so the same two inputs become INT32_MAX and 0. Clamping before the cast makes the result the same
 * everywhere.
 *
 * Out-of-range values saturate to the destination bounds and NaN maps to 0, and in-range values
 * truncate toward zero exactly as the static_cast being replaced does. That is what the goldens
 * these conversions are scored against already specify: the host branches of
 * _typecast_golden_function in ttnn/ttnn/operations/core.py truncate and then clamp for uint8,
 * uint16 and uint32, the from_torch golden in the same file clamps for uint16, and
 * tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py clamps for uint16, uint32 and
 * int32 ("Hardware saturates (clamps) values instead of wrapping on overflow").
 *
 * @tparam Int Destination integral type. Its bounds must be exactly representable as a double,
 *             which holds for every integral type up to 32 bits.
 * @tparam Float Source floating-point type.
 * @param value The value to convert.
 * @return The clamped, truncated value.
 */
template <typename Int, typename Float>
Int saturating_cast(Float value) noexcept {
    static_assert(std::is_integral_v<Int> && !std::is_same_v<Int, bool>, "saturating_cast: Int must be integral");
    static_assert(std::is_floating_point_v<Float>, "saturating_cast: Float must be floating-point");
    static_assert(
        std::numeric_limits<Int>::digits <= std::numeric_limits<double>::digits,
        "saturating_cast: the bounds of Int must round-trip through double");

    // NaN compares false against every bound, so it has to be taken out before the range test.
    if (std::isnan(value)) {
        return 0;
    }
    // Widen first: the bounds of a 32-bit destination are not all representable as a float, so
    // comparing in float would let 2^31 through as INT32_MAX and back into undefined behavior.
    const double widened = static_cast<double>(value);
    if (widened <= static_cast<double>(std::numeric_limits<Int>::lowest())) {
        return std::numeric_limits<Int>::lowest();
    }
    if (widened >= static_cast<double>(std::numeric_limits<Int>::max())) {
        return std::numeric_limits<Int>::max();
    }
    return static_cast<Int>(widened);
}

}  // namespace tt
