// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// What the index-key append's row writer (attn_key_write.cpp) decides and where it puts bytes, free of kernel-only
// dependencies so that the host compiles the same text (tests/host_key_rows.cpp, tests/test_key_rows_host.py;
// PLAN.md section 20).
//
// The key store holds BFP8 tiles: a 32-row BLOCK of one conversation's stripe rows on one chip is 4 tiles of 1,088
// bytes side by side in ONE DRAM bank (4,352 B; fused/kv_store: Kind INDEX, row_stride 0). A tile is
//     64 exponent bytes, one per face row: byte 16 f + r for row r of face f
//     4 x 256 mantissa bytes: byte 64 + 256 f + 16 r + c for column c of row r of face f (sign and 7 bits)
// with faces 0, 1 = tile rows 0..15 (columns 0..15, 16..31) and faces 2, 3 = tile rows 16..31
// (tt_metal/impl/data_format/blockfloat_common.cpp). A token row of a tile is therefore 2 exponent bytes and 2 x 16
// mantissa bytes that belong to no other row: it can be taken out of one tile and put into another.
//
// The writer patches the rows of a pass block by block: ONE read of every block that gets a row, every row of the
// block patched into that copy, ONE write. Two rows of one pass in the same block (two token rows of one
// conversation at neighbouring positions) are therefore both in the block that is written. Two rows at the SAME
// stripe row of the same block are the caller's error (one position given to two token rows of a conversation):
// the higher token row is patched last and stays, and the writer reports the pair in its status word (same_row).
#pragma once

#include <cstdint>

namespace key_rows {

constexpr uint32_t TILE_BYTES = 1088;
constexpr uint32_t TILES = 4;                         // a key is 128 wide
constexpr uint32_t BLOCK_BYTES = TILES * TILE_BYTES;  // 4,352
constexpr uint32_t BLOCK_ROWS = 32;
constexpr uint32_t NONE = 0xFFFFFFFFu;

// Byte offsets, inside a tile, of tile row `row`: its exponent byte and its 16 mantissa bytes in half h (columns
// 16 h .. 16 h + 15).
constexpr uint32_t exp_byte(uint32_t row, uint32_t h) { return 16 * (2 * (row >> 4) + h) + (row & 15); }
constexpr uint32_t man_byte(uint32_t row, uint32_t h) { return 64 + 256 * (2 * (row >> 4) + h) + 16 * (row & 15); }

// The groups of a pass. where[u] = the block of token row u as one number (NONE: the row is not written here: idle,
// another chip's, or its frame is not mapped); two token rows have the same block exactly when the numbers are
// equal. Returns the number of groups; group[u] = the index of u's group (NONE if not written), leader[g] = the
// first token row of group g (the group's block is where[leader[g]]).
template <uint32_t N>
inline uint32_t group_rows(const uint32_t (&where)[N], uint32_t (&group)[N], uint32_t (&leader)[N]) {
    uint32_t groups = 0;
    for (uint32_t u = 0; u < N; ++u) {
        group[u] = NONE;
        if (where[u] == NONE) {
            continue;
        }
        for (uint32_t g = 0; g < groups; ++g) {
            if (where[leader[g]] == where[u]) {
                group[u] = g;
                break;
            }
        }
        if (group[u] == NONE) {
            leader[groups] = u;
            group[u] = groups++;
        }
    }
    return groups;
}

// The first pair of token rows u < v of a pass that go to the same tile row of the same block, as u | v << 8; NONE
// when there is none. (row[u]: the tile row of token row u in its block; read only where where[u] != NONE.)
template <uint32_t N>
inline uint32_t same_row(const uint32_t (&where)[N], const uint32_t (&row)[N]) {
    for (uint32_t u = 0; u < N; ++u) {
        for (uint32_t v = u + 1; v < N && where[u] != NONE; ++v) {
            if (where[v] == where[u] && row[v] == row[u]) {
                return u | (v << 8);
            }
        }
    }
    return NONE;
}

// Move tile row `from` of the 4 tiles at `src` to tile row `to` of the 4 tiles at `dst` (bytes; any pointer type
// that indexes bytes).
template <typename Src, typename Dst>
inline void move_row(Src src, uint32_t from, Dst dst, uint32_t to) {
    for (uint32_t t = 0; t < TILES; ++t) {
        for (uint32_t h = 0; h < 2; ++h) {
            dst[t * TILE_BYTES + exp_byte(to, h)] = src[t * TILE_BYTES + exp_byte(from, h)];
            const uint32_t a = t * TILE_BYTES + man_byte(from, h), b = t * TILE_BYTES + man_byte(to, h);
            for (uint32_t c = 0; c < 16; ++c) {
                dst[b + c] = src[a + c];
            }
        }
    }
}

}  // namespace key_rows
