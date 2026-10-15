// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief Round integer addresses, offsets, and sizes up to power-of-two alignment boundaries.
 */

#include <type_traits>

#include <tt_stl/assert.hpp>

namespace tt {

/**
 * @brief Round addr up to the nearest multiple of a positive power of two.
 *
 * Use this to pad a buffer size or advance an address to an alignment boundary.
 * For example, align(17u, 16u) is 32, align(32u, 16u) is 32, and align(0u, 16u) is 0.
 * The address or size itself does not need to be a power of two.
 *
 * For alignment = 2^k, alignment - 1 is a mask with its lowest k bits set to one.
 * OR-ing that mask into addr - 1 sets those bits; adding one then carries into the
 * next alignment boundary. Subtracting one first keeps already-aligned values unchanged.
 * Zero remains zero; for unsigned types, the subtraction and final addition use defined wraparound.
 * This bit-mask shortcut does not round correctly for arbitrary alignment values.
 *
 * @tparam T1 Integral type of the address or size.
 * @tparam T2 Integral type of the alignment.
 * @param addr Non-negative integer address, offset, or size to round up.
 * @param alignment Positive power of two, in the same units as addr (for example, bytes).
 * @return The smallest multiple of alignment greater than or equal to addr, as
 *         std::common_type_t<T1, T2>.
 * @pre The rounded result must be representable in the return type; overflow is not checked.
 * @note TT_ASSERT checks the alignment precondition when DEBUG is defined. Callers must
 *       satisfy it in release builds too. Valid calls can be evaluated at compile time.
 */
template <typename T1, typename T2>
constexpr std::common_type_t<T1, T2> align(T1 addr, T2 alignment) {
    static_assert(std::is_integral_v<T1>, "align() requires integral types");
    static_assert(std::is_integral_v<T2>, "align() requires integral types");
    TT_ASSERT(
        alignment > 0 && (alignment & (alignment - 1)) == 0, "align() requires a positive power of two alignment");
    using T = std::common_type_t<T1, T2>;
    return ((static_cast<T>(addr) - 1) | (static_cast<T>(alignment) - 1)) + 1;
}

}  // namespace tt
