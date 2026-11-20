// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Polyfills for <type_traits> pieces from C++ versions after the one the kernels build with (C++17). Each has the
// standard name, in namespace stdx: when the build moves to that standard, replace stdx:: with std::.

#pragma once

#include <type_traits>

namespace stdx
{

/**
 * @brief T without references and cv-qualifiers (std::remove_cvref, C++20)
 *
 * @tparam T The type to strip
 */
template <class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

/// remove_cvref<T>::type (std::remove_cvref_t, C++20)
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace stdx
