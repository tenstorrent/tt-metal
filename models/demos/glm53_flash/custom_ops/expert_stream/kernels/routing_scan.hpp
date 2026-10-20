// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Active-expert scan of the routing matrix for the expert stream kernels (custom_ops/expert_stream/op.py).
//
// The routing matrix is the router's dense [1, 1, M, E] BF16 TILE tensor (token row t, expert column e holds the
// routing weight or 0): E_TILES tiles of 32x32 (M <= 32 tokens, E <= 32 * E_TILES experts). Expert e is active
// when any token routes to it. The data-movement cores read the tiles into a scratch CB and scan them; the compute
// core learns the count through the BRISC mailbox.

#pragma once

#include <stdint.h>

// Index (in BF16 elements) of element (row, col) inside one 32x32 tile: four 16x16 faces, row-major faces.
inline uint32_t tile_elem(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}

// routing: the E_TILES tiles back to back in L1. Fills `active` with the ids of the experts any of the M tokens
// routes to (ascending) and returns their count.
template <uint32_t E, uint32_t M>
inline uint32_t scan_active(const volatile tt_l1_ptr uint16_t* routing, uint16_t* active) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const volatile tt_l1_ptr uint16_t* tile = routing + (e >> 5) * 1024;
        const uint32_t c = e & 31;
        bool any = false;
        for (uint32_t t = 0; t < M && !any; ++t) {
            any = tile[tile_elem(t, c)] != 0;
        }
        if (any) {
            active[n++] = static_cast<uint16_t>(e);
        }
    }
    return n;
}
