// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Row helpers for the fused programs that work on a subset of one 32-row tile's rows (the MTP verify rows, the
// lanes): the byte offsets of a tile row in the 16 x 16 face layout and the one-hot row mask tile the recurrence
// kernels multiply with to keep an outer product's contraction to one row.  Data-movement kernels only (RISC word
// stores); bf16 tiles.  The verify-rows family (gdn_rows_scan, a future qsa_rows) shares this header.
#pragma once

#include <cstdint>

namespace fused_rows {

constexpr uint32_t TILE_ROWS = 32, FACE_ROWS = 16, FACE_BYTES = 512, HALF_ROW_BYTES = 32, FACE_ELEMS = 256;
constexpr uint32_t BF16_TILE_BYTES = 2048;

// Byte offset of the first 16 columns of tile row `row` (the second 16 columns sit FACE_BYTES further) inside a
// 32 x 32 tile of 2-byte elements: faces are 16 x 16, row-major, face (row / 16) * 2 + (col / 16).
constexpr uint32_t bf16_row_offset(uint32_t row) { return ((row >> 4) * 2) * FACE_BYTES + (row & 15) * HALF_ROW_BYTES; }

// Element index of (row, col) inside a bf16 tile (the a / b scalar patch of the GDN kernels reads it).
constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * 16 + (col & 15);
}

inline void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}

// A bf16 tile that is 1.0 on every column of tile row `row` and 0 elsewhere (gdn_step's lane mask, M_r).
inline void one_hot_row_bf16(uint32_t l1, uint32_t row) {
    fill_words(l1, BF16_TILE_BYTES / 4, 0);
    volatile tt_l1_ptr uint16_t* elems = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
    const uint32_t left = face_element(row, 0);
    for (uint32_t c = 0; c < 16; ++c) {
        elems[left + c] = 0x3F80;
        elems[left + FACE_ELEMS + c] = 0x3F80;
    }
}

// Copy tile row `row` (64 B: two 32 B face pieces) of the bf16 tile at `src` into the same row of the tile at `dst`
// with RISC word stores (16 words).
inline void copy_row_bf16(uint32_t dst, uint32_t src, uint32_t row) {
    const uint32_t offset = bf16_row_offset(row);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + offset);
    const volatile tt_l1_ptr uint32_t* s = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src + offset);
    for (uint32_t k = 0; k < HALF_ROW_BYTES / 4; ++k) {
        d[k] = s[k];
        d[k + FACE_BYTES / 4] = s[k + FACE_BYTES / 4];
    }
}

}  // namespace fused_rows
