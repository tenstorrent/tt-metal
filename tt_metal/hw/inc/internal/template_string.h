// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace internal {

// Maximum number of characters this type can hold, excluding the terminating NUL.
//
// This restriction is aribtary to support use of TemplateString type as a non-type template parameter without CTAD.
// This should go away when we upgrade to C++20.
//
// MAINTAINER: This constant must be kept in sync with MAX_ACCESSOR_NAME_LENGTH in kernel_spec.hpp on host.
inline constexpr size_t MAX_TEMPLATE_STRING_LEN = 64;

/**
 * Represents a null terminated string meant to be used as a non-type template parameter.
 *
 * Example usage:
 * ```cpp
 * template <TemplateString name>
 * int foo() {
 *   if (name == "something") {
 *     return 1;
 *   }
 *   return 0;
 * }
 *
 * static_assert(foo<TemplateString("something")>() == 1);
 * static_assert(foo<TemplateString("something else")>() == 0);
 * ```
 */
struct TemplateString {
    // invariant: always null-terminated
    char value[MAX_TEMPLATE_STRING_LEN + 1] = {};
    // Size of the string including the terminating NUL.
    std::size_t length;

    template <std::size_t N>
    constexpr TemplateString(const char (&str)[N]) : length(N) {
        static_assert(
            N <= MAX_TEMPLATE_STRING_LEN + 1,
            "TemplateString content must be at most MAX_TEMPLATE_STRING_LEN characters");
        for (size_t i = 0; i < N; ++i) {
            value[i] = str[i];
        }
    }

    constexpr bool operator==(const TemplateString& other) const {
        if (length != other.length) {
            return false;
        }
        for (size_t i = 0; i < length; ++i) {
            if (value[i] != other.value[i]) {
                return false;
            }
        }
        return true;
    }

    constexpr bool operator!=(const TemplateString& other) const { return !(*this == other); }
};

static_assert(TemplateString("something") == TemplateString("something"));
static_assert(TemplateString("something") != TemplateString("something else"));

}  // namespace internal
