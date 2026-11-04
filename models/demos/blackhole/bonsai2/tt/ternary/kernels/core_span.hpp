// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// COMMON_ARGS: every core gets the same runtime args and works out its own output tiles from its logical position
// (cores 0 .. ncores-1 in row-major order over a grid grid_x wide). Per-core args cost the dispatcher one write per
// core per kernel: a 130-core ternary linear with tiny K took 55 us per traced call.
#pragma once
#include <stdint.h>
#ifndef FORCE_INLINE
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

FORCE_INLINE uint32_t core_index(uint32_t grid_x) {
    return (uint32_t)get_absolute_logical_y() * grid_x + (uint32_t)get_absolute_logical_x();
}

// Output tiles [first, first + n) of this core: Nt tiles in units of `unit` tiles, split as evenly as possible with
// the first (Nt / unit) % ncores cores taking one unit more (the host's split).
FORCE_INLINE void core_span(
    uint32_t Nt, uint32_t ncores, uint32_t grid_x, uint32_t unit, uint32_t& first, uint32_t& n) {
    const uint32_t c = core_index(grid_x);
    const uint32_t units = Nt / unit, base = units / ncores, extra = units % ncores;
    n = unit * (base + (c < extra ? 1 : 0));
    first = unit * (c * base + (c < extra ? c : extra));
}
