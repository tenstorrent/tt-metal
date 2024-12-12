// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

inline __attribute__((always_inline)) uint32_t align(uint32_t addr, uint32_t alignment) {
    uint32_t remainder = addr % alignment;
    if (remainder == 0) {
        return addr;
    }
    return addr + (alignment - remainder);
}
