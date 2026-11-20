// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Where the pieces of the index query are (fused/layer_attn/PLAN.md section 21.3), free of kernel-only dependencies
// so that the host compiles the same text (tests/host_idx_rows.cpp, tests/test_idx_rows_host.py).
//
// The landing on the idx core: chip h's 4 row tiles of index head h's query for the pass's token rows, as the pre
// group leaves them: [chip h][tile t][half f][token row u], 16 BF16 values (32 B) each: dims 32 t + 16 f .. of head
// h for token row u. A row tile of `rows` token rows is 64 * rows bytes.
// A user's query for the scan (ScanConfig(query_rows=True)): 4 standard BF16 tiles [32 heads x 32 dims], tile t the
// dims 32 t ..; a standard tile is 4 faces of [16 x 16]: face 2 (h / 16) + f holds, in its row h % 16, the 16 values
// of head h's dims 32 t + 16 f ..: the same 32 B. So a user's 4 tiles are 32 x 4 x 2 = 256 pieces moved whole.
// The head weights: chip 22's pool tile, a row tile [token rows x 32 heads]; token row u's 2 x 32 B are, in this
// order, the 32 BF16 multipliers the scan's step page holds in its words 16..31 (bytes 64..127).
#pragma once

#include <cstdint>

namespace idx_rows {

constexpr uint32_t HEADS = 32;
constexpr uint32_t TILES = 4;                         // 128 dims
constexpr uint32_t PIECE_WORDS = 8;                   // 16 BF16 values
constexpr uint32_t TILE_WORDS = 512;                  // a standard BF16 tile
constexpr uint32_t QUERY_WORDS = TILES * TILE_WORDS;  // a user's query: 8,192 B
constexpr uint32_t STEP_MULTIPLIER_WORD = 16;         // of the scan's step page

// Word offsets. `rows`: token rows of a row tile (8 or 16).
constexpr uint32_t row_tile_words(uint32_t rows) { return 16 * rows; }
constexpr uint32_t landing_words(uint32_t rows) { return HEADS * TILES * row_tile_words(rows); }
constexpr uint32_t src_word(uint32_t rows, uint32_t h, uint32_t t, uint32_t f, uint32_t u) {
    return (h * TILES + t) * row_tile_words(rows) + f * PIECE_WORDS * rows + u * PIECE_WORDS;
}
constexpr uint32_t dst_word(uint32_t h, uint32_t t, uint32_t f) {
    return t * TILE_WORDS + (2 * (h >> 4) + f) * (TILE_WORDS / 4) + (h & 15) * PIECE_WORDS;
}
constexpr uint32_t weight_word(uint32_t rows, uint32_t f, uint32_t u) {
    return f * PIECE_WORDS * rows + u * PIECE_WORDS;
}

// Token row u's 256 pieces: f(word offset of the piece in the landing, its word offset in the query's 4 tiles).
template <typename F>
inline void pieces(uint32_t rows, uint32_t u, F f) {
    for (uint32_t h = 0; h < HEADS; ++h) {
        for (uint32_t t = 0; t < TILES; ++t) {
            for (uint32_t half = 0; half < 2; ++half) {
                f(src_word(rows, h, t, half, u), dst_word(h, t, half));
            }
        }
    }
}

// Token row u's query: landing (words) -> 4 standard tiles (words).
template <typename Src, typename Dst>
inline void assemble(Src landing, uint32_t rows, uint32_t u, Dst query) {
    pieces(rows, u, [&](uint32_t a, uint32_t b) {
        for (uint32_t i = 0; i < PIECE_WORDS; ++i) {
            query[b + i] = landing[a + i];
        }
    });
}

// Token row u's 32 multipliers: the head-weight row tile (words) -> the 16 words of the step page.
template <typename Src, typename Dst>
inline void multipliers(Src weights, uint32_t rows, uint32_t u, Dst words) {
    for (uint32_t f = 0; f < 2; ++f) {
        for (uint32_t i = 0; i < PIECE_WORDS; ++i) {
            words[f * PIECE_WORDS + i] = weights[weight_word(rows, f, u) + i];
        }
    }
}

}  // namespace idx_rows
