// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

inline __attribute__((always_inline)) constexpr uint32_t align_power_of_2(uint32_t addr, uint32_t alignment) {
    return ((addr - 1) | (alignment - 1)) + 1;
}

inline __attribute__((always_inline)) constexpr uint32_t align(uint32_t addr, uint32_t alignment) {
    return ((addr + alignment - 1) / alignment) * alignment;
}

inline __attribute__((always_inline)) constexpr bool is_power_of_2(uint32_t n) {
    return (n != 0) && (n & (n - 1)) == 0;
}
