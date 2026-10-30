// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Row helpers for the fused programs that work on a subset of one 32-row tile's rows (the MTP verify rows, the
// lanes): the byte offsets of a tile row in the 16 x 16 face layout and the one-hot row mask tile the recurrence
// kernels multiply with to keep an outer product's contraction to one row.  Data-movement kernels only (RISC word
// stores); bf16 tiles.
#pragma once

#include <cstdint>

#include "api/core_local_mem.h"

namespace fused_rows {

constexpr uint32_t tile_rows = 32, face_rows = 16, face_bytes = 512, half_row_bytes = 32, face_elements = 256;
constexpr uint32_t bf16_tile_bytes = 2048;

// Byte offset of the first 16 columns of tile row `row` (the second 16 columns sit face_bytes further) inside a
// 32 x 32 tile of 2-byte elements: faces are 16 x 16, row-major, face (row / 16) * 2 + (col / 16).
constexpr uint32_t bf16_row_offset(uint32_t row) { return ((row >> 4) * 2) * face_bytes + (row & 15) * half_row_bytes; }

// Element index of (row, col) inside a bf16 tile (the a / b scalar patch of the GDN kernels reads it).
constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * 16 + (col & 15);
}

inline void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = value;
    }
}

// A bf16 tile that is 1.0 on every column of tile row `row` and 0 elsewhere (gdn_step's lane mask, M_r).
inline void one_hot_row_bf16(uint32_t l1_addr, uint32_t row) {
    fill_words(l1_addr, bf16_tile_bytes / 4, 0);
    CoreLocalMem<volatile uint16_t> elements(l1_addr);
    const uint32_t left = face_element(row, 0);
    for (uint32_t c = 0; c < 16; ++c) {
        elements[left + c] = 0x3F80;
        elements[left + face_elements + c] = 0x3F80;
    }
}

// Copy tile row `row` (64 B: two 32 B face pieces) of the bf16 tile at `src_addr` into the same row of the tile at
// `dst_addr` with RISC word stores (16 words).
inline void copy_row_bf16(uint32_t dst_addr, uint32_t src_addr, uint32_t row) {
    const uint32_t offset = bf16_row_offset(row);
    CoreLocalMem<volatile uint32_t> dst(dst_addr + offset);
    CoreLocalMem<const volatile uint32_t> src(src_addr + offset);
    for (uint32_t k = 0; k < half_row_bytes / 4; ++k) {
        dst[k] = src[k];
        dst[k + face_bytes / 4] = src[k + face_bytes / 4];
    }
}

}  // namespace fused_rows
