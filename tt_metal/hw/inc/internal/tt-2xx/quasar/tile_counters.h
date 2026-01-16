// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Adds tile counters base array + struct definition

#ifndef _TILE_COUNTERS_H_
#define _TILE_COUNTERS_H_

#define TILE_COUNTERS_BASE             0x0080c000

constexpr uint32_t NUM_WORDS_TILE_CNT = 8;
typedef struct {
    std::uint32_t reserved0     : 32;
    std::uint32_t reset         : 32;
    std::uint32_t posted        : 32;
    std::uint32_t acked         : 32;
    std::uint32_t buf_capacity  : 32;
    std::uint32_t tiles_posted_raw : 32;
    std::uint32_t tiles_acked_raw  : 32;
} tile_counter_t;

static_assert(sizeof(tile_counter_t) == NUM_WORDS_TILE_CNT*sizeof(uint32_t), "tile_counter_t must be 96b!");

typedef union {
    uint32_t words[NUM_WORDS_TILE_CNT];
    tile_counter_t f;
} tile_counter_u;

tile_counter_u volatile * const tile_counters = (tile_counter_u volatile * const) TILE_COUNTERS_BASE;

#endif  // ifndef _TILE_COUNTERS_H_
