// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>

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
constexpr uint32_t div_up(uint32_t a, uint32_t b) {
    assert(b != 0 && "Divide by zero error in div_up");
    return static_cast<uint32_t>((a + b - 1) / b);
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
constexpr uint32_t round_up(uint32_t a, uint32_t b) { return b * div_up(a, b); }

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
constexpr uint32_t round_down(uint32_t a, uint32_t b) { return a / b * b; }

}  // namespace tt
