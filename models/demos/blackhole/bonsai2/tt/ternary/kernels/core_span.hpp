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
#ifdef RX_INFO
    // the decode weight stream's receiver layout (bonsai2/tt/stream/stream.py): this core's receiver index
    (void)grid_x;
    return reinterpret_cast<volatile uint32_t*>(RX_INFO)[0];
#else
    return (uint32_t)get_absolute_logical_y() * grid_x + (uint32_t)get_absolute_logical_x();
#endif
}

// Output tiles [first, first + n) of this core: Nt tiles in units of `unit` tiles, split as evenly as possible with
// the first (Nt / unit) % ncores cores taking one unit more (the host's split).
FORCE_INLINE void core_span(
    uint32_t Nt, uint32_t ncores, uint32_t grid_x, uint32_t unit, uint32_t& first, uint32_t& n) {
    uint32_t c = core_index(grid_x);
#ifdef RX_SPAN_KEYS
    // the weight stream's sender-balanced split for ops on ncores receivers over Nt / unit units (stream.py
    // span_keys: info words 4 .., first unit << 16 | units)
    constexpr uint32_t span_keys[] = RX_SPAN_KEYS;
    const uint32_t key = (ncores << 16) | (Nt / unit);
    for (uint32_t k = 0; k < sizeof(span_keys) / sizeof(span_keys[0]); ++k) {
        if (span_keys[k] == key) {
            const uint32_t v = reinterpret_cast<volatile uint32_t*>(RX_INFO)[4 + k];
            first = unit * (v >> 16);
            n = unit * (v & 0xffff);
            return;
        }
    }
#endif
#ifdef CORE_OFFSET
    c -= CORE_OFFSET;  // this op on cores CORE_OFFSET .. (another op's kernels share its program on the cores before)
#endif
    const uint32_t units = Nt / unit, base = units / ncores, extra = units % ncores;
    n = unit * (base + (c < extra ? 1 : 0));
    first = unit * (c * base + (c < extra ? c : extra));
}
