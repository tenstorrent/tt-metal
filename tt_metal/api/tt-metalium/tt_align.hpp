// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief Round integer addresses, offsets, and sizes up to power-of-two alignment boundaries.
 */

#include <cassert>
#include <type_traits>

namespace tt {

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
 *      If NDEBUG is not defined, an assert stops the program when alignment is not valid.
 *      If NDEBUG is defined, the function does not do this check, and the behavior is not defined.
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
