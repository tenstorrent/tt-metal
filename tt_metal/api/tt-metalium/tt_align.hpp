// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt {

template <typename A, typename B>
inline constexpr auto align(A addr, B alignment) {
    if constexpr (std::is_pointer_v<A>) {
        using P = A;  // Preserve pointer type
        using T = std::common_type_t<std::uintptr_t, B>;

        std::uintptr_t addr_int = reinterpret_cast<std::uintptr_t>(addr);
        T result = ((static_cast<T>(addr_int) - 1) | (static_cast<T>(alignment) - 1)) + 1;

        return reinterpret_cast<P>(static_cast<std::uintptr_t>(result));
    } else {
        using T = std::common_type_t<A, B>;
        return ((static_cast<T>(addr) - 1) | (static_cast<T>(alignment) - 1)) + 1;
    }
}

}  // namespace tt
