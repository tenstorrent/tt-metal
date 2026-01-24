// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifndef COMPILE_FOR_TRISC
#include <overlay/meta/registers/overlay_reg_defines_core.h>
#endif

// Adds tile counters base array + struct definition

#ifndef _TILE_COUNTERS_H_
#define _TILE_COUNTERS_H_

#ifndef COMPILE_FOR_TRISC
#define LOCAL_TILE_COUNTERS_BASE TT_OVERLAY_LLK_TILE_COUNTERS_REG_MAP_BASE_ADDR
#else
#define LOCAL_TILE_COUNTERS_BASE TILE_COUNTERS_BASE
#endif

constexpr uint32_t NUM_TILE_COUNTERS = 16;
constexpr uint32_t NUM_WORDS_TILE_CNT = 16;

typedef struct {
    std::uint32_t reserved0;
    std::uint32_t reset;
    std::uint32_t posted;
    std::uint32_t acked;
    std::uint32_t buf_capacity;
    std::uint32_t tiles_posted_raw;
    std::uint32_t tiles_acked_raw;
    std::uint32_t error_status;
    std::uint32_t reserved1[8];
} tile_counter_t;

static_assert(
    sizeof(tile_counter_t) == NUM_WORDS_TILE_CNT * sizeof(uint32_t), "tile_counter_t must be 8 words (32 bytes)!");

typedef union {
    uint32_t words[NUM_WORDS_TILE_CNT];
    tile_counter_t f;
} tile_counter_u;

extern volatile tile_counter_u* const tile_counters;

inline void tile_counters_reset() {
    for (uint32_t i = 0; i < NUM_TILE_COUNTERS; i++) {
        tile_counters[i].f.reset = 1;
        tile_counters[i].f.buf_capacity = 1;
    }
}

#endif  // ifndef _TILE_COUNTERS_H_
