// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

// Ceiling division: the smallest integer >= a / b (`b` non-zero). The quotient plus a carry rather than
// (a + b - 1) / b, which wraps when a + b - 1 overflows the type; the same form as tt::div_up on the host. A
// template, so a kernel that declares its own non-template div_up keeps compiling: for arguments that match
// that declaration exactly the non-template wins, otherwise this one is used.
template <typename A, typename B>
__attribute__((always_inline)) constexpr auto div_up(A a, B b) noexcept -> std::common_type_t<A, B> {
    using T = std::common_type_t<A, B>;
    const T numerator = static_cast<T>(a);
    const T denominator = static_cast<T>(b);
    const T quotient = numerator / denominator;
    const T remainder = numerator % denominator;
    if constexpr (std::is_signed_v<T>) {
        return quotient + static_cast<T>(remainder != 0 && (remainder > 0) == (denominator > 0));
    }
    return quotient + static_cast<T>(remainder != 0);
}

__attribute__((always_inline)) constexpr uint32_t align_power_of_2(uint32_t addr, uint32_t alignment) {
    return ((addr - 1) | (alignment - 1)) + 1;
}

__attribute__((always_inline)) constexpr uint32_t align(uint32_t addr, uint32_t alignment) {
    return div_up(addr, alignment) * alignment;
}

__attribute__((always_inline)) constexpr bool is_power_of_2(uint32_t n) { return (n != 0) && (n & (n - 1)) == 0; }
