// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Polyfills for <algorithm> pieces from C++ versions after the one the kernels build with (C++17). Each has the
// standard name, in namespace stdx: when the build moves to that standard, replace stdx:: with std::.

#pragma once

#include <iterator>

namespace stdx
{

/**
 * @brief Check if two iterator ranges have the same length and equal elements (std::equal, constexpr from C++20)
 *
 * @tparam It1 The iterator type of the first range
 * @tparam It2 The iterator type of the second range
 * @param first1 The start of the first range
 * @param last1 The end of the first range
 * @param first2 The start of the second range
 * @param last2 The end of the second range
 * @return true if both ranges have the same number of elements and each element equals the one at its place
 */
template <class It1, class It2>
constexpr bool equal(It1 first1, It1 last1, It2 first2, It2 last2)
{
    for (; first1 != last1 && first2 != last2; ++first1, ++first2)
    {
        if (!(*first1 == *first2))
        {
            return false;
        }
    }
    return first1 == last1 && first2 == last2;
}

namespace ranges
{

/**
 * @brief Check if two ranges have the same length and equal elements (std::ranges::equal, C++20)
 *
 * Usable in a constant expression, which operator== of std::array is only from C++20.
 *
 * @code
 * stdx::ranges::equal(std::array {1, 2}, std::array {1, 2})   // true
 * @endcode
 *
 * @tparam R1 The type of the first range
 * @tparam R2 The type of the second range
 * @param a The first range
 * @param b The second range
 * @return true if a and b have the same number of elements and each element of a equals the one at its place in b
 */
template <class R1, class R2>
constexpr bool equal(const R1& a, const R2& b)
{
    return stdx::equal(std::begin(a), std::end(a), std::begin(b), std::end(b));
}

} // namespace ranges

} // namespace stdx
