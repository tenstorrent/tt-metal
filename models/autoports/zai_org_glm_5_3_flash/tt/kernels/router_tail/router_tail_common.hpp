// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused router tail (Stage 7a, router_tail.py is the host side): one core per token row t (logical (t % 8, t / 8)).
// The row's 10 logit column tiles are packed into one tile (packed row r = column tile r, i.e. experts 32r .. 32r+31),
// scores = sigmoid(logits), choice = scores + bias (the selection bias, -1e9 on the padded experts 288..319), the
// RISC picks the top-8 of the 288 choices (integer keys of the fp32 bits, descending, ties to the lower expert id) and
// writes the ids row [8] uint16 and the bf16 scores row [288].
#pragma once

#include <cstdint>

#ifndef RTAIL_CB0
#define RTAIL_CB0 0  // megakernel: this phase's CB indices shifted past the other phases'
#endif

namespace rtail {

constexpr uint32_t COL_TILES = 10;  // 320 padded router columns
#ifndef RTAIL_KSPLIT
#define RTAIL_KSPLIT 1  // Stage 14: the router's K-chunk partials side by side (chunk c at column tile c * COL_TILES)
#endif
constexpr uint32_t N_EXPERTS = 288;
constexpr uint32_t SCORE_TILES = 9;  // 288 / 32
constexpr uint32_t TOP_K = 8;

// RTAIL_CB_MAP (the PMK P12 tail, W7 step 3): each index from its own define (the fused member reuses idle indices of
// the matching format: LOG / BIAS on UnpackToDestFp32 CBs, SCORES on a bf16 one), else RTAIL_CB0 + 0 .. 4
#if defined(RTAIL_CB_MAP)
constexpr uint32_t CB_LOG = RTAIL_CB_LOG;
constexpr uint32_t CB_BIAS = RTAIL_CB_BIAS;
constexpr uint32_t CB_CHOICE = RTAIL_CB_CHOICE;
constexpr uint32_t CB_SCORES = RTAIL_CB_SCORES;
constexpr uint32_t CB_IDX = RTAIL_CB_IDX;
#else
constexpr uint32_t CB_LOG = RTAIL_CB0 + 0;     // fp32 packed logits
constexpr uint32_t CB_BIAS = RTAIL_CB0 + 1;    // fp32 packed bias (host-built, same packing)
constexpr uint32_t CB_CHOICE = RTAIL_CB0 + 2;  // fp32 packed choice
constexpr uint32_t CB_SCORES = RTAIL_CB0 + 3;  // bf16 packed scores (rounded like ttnn.typecast)
constexpr uint32_t CB_IDX = RTAIL_CB0 + 4;     // writer scratch: the 8 ids (16 B)
#endif

// RTAIL_ROWTILES (prefill, C > 32 rows): the logits are [C, 320 x KSPLIT] fp32 TILE -> row t lives in tile row t / 32
// (pages (t / 32) x COL_TILES x KSPLIT + ..), at row t % 32 inside the tile
#if defined(RTAIL_ROWTILES)
constexpr uint32_t log_page(uint32_t t, uint32_t col_tile) { return (t >> 5) * COL_TILES * RTAIL_KSPLIT + col_tile; }
constexpr uint32_t row_in_tile(uint32_t t) { return t & 31u; }
#else
constexpr uint32_t log_page(uint32_t /*t*/, uint32_t col_tile) { return col_tile; }
constexpr uint32_t row_in_tile(uint32_t t) { return t; }
#endif

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

// the row argument: first token row in bits 0..15, (rows - 1) in bits 16..31 (one row: the row itself); a
// megakernel core without a row gets 0xFFFFFFFF
constexpr uint32_t ROW_SKIP = 0xFFFFFFFFu;
constexpr uint32_t first_row(uint32_t arg) { return arg & 0xFFFFu; }
constexpr uint32_t row_count(uint32_t arg) { return (arg >> 16) + 1u; }

// monotonic unsigned key of an fp32 bit pattern (larger float -> larger key; -0 < +0, NaN never produced here)
constexpr uint32_t sort_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

}  // namespace rtail
