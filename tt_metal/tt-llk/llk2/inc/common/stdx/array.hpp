// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Polyfills for <array> pieces from C++ versions after the one the kernels build with (C++17). Each has the standard
// name, in namespace stdx: when the build moves to that standard, replace stdx:: with std::.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace stdx
{

namespace detail
{

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&list)[N], std::index_sequence<I...>)
{
    return {{list[I]...}};
}

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&&list)[N], std::index_sequence<I...>)
{
    return {{std::move(list[I])...}};
}

} // namespace detail

/**
 * @brief A std::array with a copy of each element of a C array (std::to_array, C++20)
 *
 * @tparam T The element type
 * @tparam N The number of elements
 * @param list The C array to copy from
 * @return The elements of list, in order
 */
template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&list)[N])
{
    return detail::to_array(list, std::make_index_sequence<N> {});
}

/**
 * @brief A std::array with the elements of a temporary C array or a braced list moved in (std::to_array, C++20)
 *
 * @code
 * stdx::to_array({1, 2, 3})   // std::array<int, 3>
 * @endcode
 *
 * @tparam T The element type
 * @tparam N The number of elements
 * @param list The C array or braced list to move from
 * @return The elements of list, in order
 */
template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&&list)[N])
{
    return detail::to_array(std::move(list), std::make_index_sequence<N> {});
}

} // namespace stdx
