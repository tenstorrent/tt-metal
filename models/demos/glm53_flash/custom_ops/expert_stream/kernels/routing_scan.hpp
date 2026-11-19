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

// ---------------------------------------------------------------------------------------------- E1 additions
// (decode roofline action E1, 2026-10-26; custom_ops/expert_stream/op.py ``routing_fold`` / ``scan_words``.) Both are
// compiled in only under their preprocessor defines (ROUTING_FOLD / SCAN_WORDS); the default programs keep scan_active.

// Routing-tile window of the selector fold: the kernels read the router's FULL [M, E_total] routing matrix and scan
// the E columns [col_base, col_base + E) of this chip's experts themselves (the per-column selector matmul of
// tt/moe.py `_block_stream` is a 0/1 identity block: reading the columns is bit-identical to multiplying by it). The
// window starts at tile `col_base / 32` and spans `fold_tiles` tiles; the first local expert sits at column `col_off`
// of the window's first tile. R_TILES (compile-time) = the worst-case window, (E + 62) / 32 tiles.
inline uint32_t fold_first_tile(uint32_t col_base) { return col_base >> 5; }
inline uint32_t fold_col_off(uint32_t col_base) { return col_base & 31; }
template <uint32_t E>
inline uint32_t fold_tiles(uint32_t col_base) {
    return (fold_col_off(col_base) + E + 31) >> 5;
}

// Word-wise active-expert scan (SCAN_WORDS): the same answer as scan_active for the window described above, read as
// 32-bit words per 16-column face row (8 words, OR-ed: a zero face row -- the common case, a token routes to ~2 of
// the chip's 72 experts -- costs 8 loads instead of 16 element loads with index arithmetic; a non-zero one marks its
// columns in a per-tile 32-bit mask). `routing` = `n_tiles` tiles back to back in L1 (the window); experts are
// enumerated ascending from the masks. ~1536 word loads at E = 72, M = 32 against ~2304 scalar loads + the
// tile_elem arithmetic of scan_active (the per-launch prelude of every stream kernel, before any weight is read).
template <uint32_t E, uint32_t M, uint32_t R_TILES>
inline uint32_t scan_active_words(
    const volatile tt_l1_ptr uint16_t* routing, uint16_t* active, uint32_t col_off, uint32_t n_tiles) {
    uint32_t mask[R_TILES];
    for (uint32_t i = 0; i < R_TILES; ++i) {
        mask[i] = 0;
    }
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const volatile tt_l1_ptr uint32_t* tile =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(routing + i * 1024);
        uint32_t m = 0;
        for (uint32_t t = 0; t < M; ++t) {
            const uint32_t w0 = ((t >> 4) * 2) * 128 + (t & 15) * 8;  // word index of face row t, columns 0..15
            for (uint32_t f = 0; f < 2; ++f) {                        // f = 1: columns 16..31 (the next face)
                const volatile tt_l1_ptr uint32_t* row = tile + w0 + f * 128;
                const uint32_t any = row[0] | row[1] | row[2] | row[3] | row[4] | row[5] | row[6] | row[7];
                if (any == 0) {
                    continue;
                }
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t v = row[j];  // elements 2j (low half) and 2j + 1 (high half) of the face row
                    if (v & 0xFFFFu) {
                        m |= 1u << (f * 16 + 2 * j);
                    }
                    if (v >> 16) {
                        m |= 1u << (f * 16 + 2 * j + 1);
                    }
                }
            }
        }
        mask[i] = m;
    }
    uint32_t n = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t c = col_off + e;
        if ((mask[c >> 5] >> (c & 31)) & 1u) {
            active[n++] = static_cast<uint16_t>(e);
        }
    }
    return n;
}

// Element index of (row t, local expert e) in the routing window (the D_e diagonal build and any weight lookup).
inline uint32_t fold_elem(uint32_t t, uint32_t e, uint32_t col_off) {
    const uint32_t c = col_off + e;
    return (c >> 5) * 1024 + tile_elem(t, c & 31);
}

// The classic element scan over the fold window (ROUTING_FOLD without SCAN_WORDS): scan_active with the column offset.
template <uint32_t E, uint32_t M>
inline uint32_t scan_active_fold(const volatile tt_l1_ptr uint16_t* routing, uint16_t* active, uint32_t col_off) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < E; ++e) {
        bool any = false;
        for (uint32_t t = 0; t < M && !any; ++t) {
            any = routing[fold_elem(t, e, col_off)] != 0;
        }
        if (any) {
            active[n++] = static_cast<uint16_t>(e);
        }
    }
    return n;
}
