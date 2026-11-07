// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Row addressing inside a 32x32 bf16 tile (faces 0/1 hold rows 0-15, 2/3 rows 16-31; a row is two 32-byte chunks,
// half_row_bytes each). The RISC copies are volatile word / element accesses through CoreLocalMem.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"

namespace tile_rows {
constexpr uint32_t tile_bytes = 2048, face_bytes = 512, half_row_bytes = 32, rows_per_tile = 32;

constexpr uint32_t chunk_offset(uint32_t row, uint32_t half) {
    return (((row >> 4) << 1) + half) * face_bytes + (row & 15) * half_row_bytes;
}

// row `src_row` of the tile at `src` -> row `dst_row` of the tile at `dst`; -0.0 becomes +0.0 when `canonicalize`
inline void copy_tile_row(uint32_t src, uint32_t src_row, uint32_t dst, uint32_t dst_row, bool canonicalize) {
    for (uint32_t half = 0; half < 2; ++half) {
        const CoreLocalMem<const volatile uint16_t> src_chunk(src + chunk_offset(src_row, half));
        const CoreLocalMem<volatile uint16_t> dst_chunk(dst + chunk_offset(dst_row, half));
        for (uint32_t k = 0; k < 16; ++k) {
            const uint16_t element = src_chunk[k];
            dst_chunk[k] = (canonicalize && element == 0x8000) ? 0 : element;
        }
    }
}

// a ROW_MAJOR row of `tiles` * 32 bf16 at `row_l1` -> row `dst_row` of the `tiles` consecutive tiles at `tiles_l1`
inline void place_row(uint32_t row_l1, uint32_t tiles_l1, uint32_t tiles, uint32_t dst_row) {
    for (uint32_t t = 0; t < tiles; ++t) {
        for (uint32_t half = 0; half < 2; ++half) {
            const CoreLocalMem<const volatile uint32_t> src_chunk(
                row_l1 + t * 2 * half_row_bytes + half * half_row_bytes);
            const CoreLocalMem<volatile uint32_t> dst_chunk(tiles_l1 + t * tile_bytes + chunk_offset(dst_row, half));
            for (uint32_t k = 0; k < half_row_bytes / 4; ++k) {
                dst_chunk[k] = src_chunk[k];
            }
        }
    }
}

inline void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    const CoreLocalMem<volatile uint32_t> dst_words(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst_words[k] = value;
    }
}
}  // namespace tile_rows
