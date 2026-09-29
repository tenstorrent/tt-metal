// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Row addressing inside a 32x32 bf16 tile (faces 0/1 hold rows 0-15, 2/3 rows 16-31; a row is two 32-byte chunks).

#pragma once

#include <cstdint>

namespace tile_rows {
constexpr uint32_t TILE_BYTES = 2048, FACE_BYTES = 512, ROW_BYTES = 32, TILE_ROWS = 32;

constexpr uint32_t chunk_offset(uint32_t row, uint32_t half) {
    return (((row >> 4) << 1) + half) * FACE_BYTES + (row & 15) * ROW_BYTES;
}

// row `src_row` of the tile at `src` -> row `dst_row` of the tile at `dst`, -0.0 canonicalized to +0.0 when `canon`
inline void copy_tile_row(uint32_t src, uint32_t src_row, uint32_t dst, uint32_t dst_row, bool canon) {
    for (uint32_t half = 0; half < 2; ++half) {
        volatile tt_l1_ptr uint16_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src + chunk_offset(src_row, half));
        volatile tt_l1_ptr uint16_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + chunk_offset(dst_row, half));
        for (uint32_t k = 0; k < 16; ++k) {
            const uint16_t v = s[k];
            d[k] = (canon && v == 0x8000) ? 0 : v;
        }
    }
}

// a ROW_MAJOR row of `tiles` * 32 bf16 at `row_l1` -> row `dst_row` of the `tiles` consecutive tiles at `tiles_l1`
inline void place_row(uint32_t row_l1, uint32_t tiles_l1, uint32_t tiles, uint32_t dst_row) {
    for (uint32_t t = 0; t < tiles; ++t) {
        for (uint32_t half = 0; half < 2; ++half) {
            volatile tt_l1_ptr uint32_t* s =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1 + t * 2 * ROW_BYTES + half * ROW_BYTES);
            volatile tt_l1_ptr uint32_t* d =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tiles_l1 + t * TILE_BYTES + chunk_offset(dst_row, half));
            for (uint32_t k = 0; k < ROW_BYTES / 4; ++k) {
                d[k] = s[k];
            }
        }
    }
}

inline void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}
}  // namespace tile_rows
