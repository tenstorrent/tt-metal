// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt {

template <typename T1, typename T2>
inline constexpr std::common_type_t<T1, T2> align(T1 addr, T2 alignment) {
    static_assert(std::is_integral<T1>::value, "align() requires integral types");
    static_assert(std::is_integral<T2>::value, "align() requires integral types");
    using T = std::common_type_t<T1, T2>;
    return ((static_cast<T>(addr) - 1) | (static_cast<T>(alignment) - 1)) + 1;
}

}  // namespace tt
