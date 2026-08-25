// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _TIMESTAMP_HPP
#define _TIMESTAMP_HPP

#define WALL_CLOCK_L (*(volatile uint32_t*)0xFFB121F0)
#define WALL_CLOCK_H (*(volatile uint32_t*)0xFFB121F8)

static inline uint64_t timestamp() {
    uint64_t low = WALL_CLOCK_L;
    asm volatile("" ::: "memory");
    uint64_t high = WALL_CLOCK_H;

    return (high << 32) | low;
}

#endif /* _TIMESTAMP_HPP */
