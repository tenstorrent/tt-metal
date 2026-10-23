// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_tiled: constants and message layouts shared by the reader, writer and compute kernels.  The Python
// side (ttnn/fused/sparse_sdpa_tiled/__init__.py) mirrors MSG_WORDS / the field order; the static test pins both.
//
// Work unit = one tile of TQ consecutive query rows x all H local heads (R = TQ * H rows, (head, query) order:
// row h * TQ + q).  The reader builds the tile's block union (seed prefix, then ascending block ids) and the
// per-slot membership words; the K/V rows of the union stream once per tile in chunks of CB blocks; the writer
// turns the membership words into one additive bf16 mask band per chunk (0 = attended, MASK_FLOOR = hidden);
// compute runs the streaming flash loop with the band added to the scores before the running max.

#pragma once

#include <stdint.h>

namespace sst {

constexpr uint32_t TILE_HW = 1024;
constexpr uint32_t FACE_HW = 256;
constexpr uint32_t FACE_W = 16;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t BF16_TILE_BYTES = 2048;

// The most negative finite bf16 (0xFF7F = -3.3895e38): the mask value.  Finite, so a score plus the mask, the
// chunk max of a memberless row and the correction argument stay finite (the seed block in chunk 0 keeps every
// row's running max a real score; a memberless chunk leaves it unchanged and its probabilities exp(-2.1e37) = 0).
constexpr uint32_t MASK_FLOOR_BF16 = 0xFF7Fu;
constexpr uint32_t MASK_FLOOR_PAIR = 0xFF7FFF7Fu;

constexpr uint32_t NO_BLOCK = 0xFFFFFFFFu;

// One 16-byte page per message; word indices below.
constexpr uint32_t MSG_WORDS = 4;
constexpr uint32_t MSG_PAGE_BYTES = MSG_WORDS * sizeof(uint32_t);

// reader -> compute, one page per tile
namespace ctrl {
constexpr uint32_t N_CHUNKS = 0;
}  // namespace ctrl

// reader -> writer, one page per tile
namespace tilectl {
constexpr uint32_t U = 0;         // union slots
constexpr uint32_t N_CHUNKS = 1;  // ceil(U / CB)
constexpr uint32_t T0 = 2;        // first query row of the tile
constexpr uint32_t N_SEEDS = 3;   // the seed prefix's length (slots [0, N_SEEDS) ascending, [N_SEEDS, U) ascending)
}  // namespace tilectl

// reader -> writer, one page per chunk (the dual-NoC gather request, as sparse_sdpa's)
namespace kreq {
constexpr uint32_t SLOT0 = 0;    // first union slot of the chunk
constexpr uint32_t SPLIT = 1;    // the writer gathers chunk rows [0, SPLIT), the reader [SPLIT, valid)
constexpr uint32_t IS_LAST = 2;  // last chunk of the tile
constexpr uint32_t DST_L1 = 3;   // the reserved cb_k_rm write pointer
}  // namespace kreq

FORCE_INLINE uint32_t popcount32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24;
}

// Index of the lowest set bit (v != 0).
FORCE_INLINE uint32_t ctz32(uint32_t v) { return popcount32((v & (0u - v)) - 1u); }

// Element offset of (row, col) inside a 32x32 tile stored as four 16x16 faces (row-major faces 0..3 = top-left,
// top-right, bottom-left, bottom-right; each face row-major).
FORCE_INLINE uint32_t tile_elem(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_HW + (row & 15) * FACE_W + (col & 15);
}

}  // namespace sst
