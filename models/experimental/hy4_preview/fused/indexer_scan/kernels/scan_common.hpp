// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the three kernels of the index-key scan (reader, compute, writer) and by the cut kernel.
//
// One "scan core" serves one (user, DRAM bank) pair of one chip: it streams that bank's share of the user's
// index-key stripe, scores it, keeps its best 2048 (score, position) pairs and takes part in a binary tree over
// the user's 8 banks. Every scan core has the same circular buffers, so a CB has the same L1 address on all of
// them: that is what lets a core write its partner's landing CB.

#pragma once

#include <cstdint>

namespace hy4_scan {

// Circular buffers.
constexpr uint32_t cb_k = 0;        // index keys, BFP8 tiles, 4 per 32-position block (reader -> compute)
constexpr uint32_t cb_qt = 1;       // the query, [128 dims x 32 heads] as 4 BF16 tiles (reader -> compute)
constexpr uint32_t cb_sel = 2;      // 32 head-sum selector tiles (reader -> compute)
constexpr uint32_t cb_mask = 3;     // 2 BF16 tiles masking the tile the context ends in (reader -> compute)
constexpr uint32_t cb_meta = 4;     // one 64-byte page of uint32 words, below (reader -> compute)
constexpr uint32_t cb_landing = 5;  // a partner's fused survivor, 2 raw 32-bit tiles (writer -> compute)
constexpr uint32_t cb_centre = 6;   // 2 BF16 tiles that subtract the user's score centre (reader -> compute)
constexpr uint32_t cb_p = 16;       // relu(key . query head): [32 positions x 32 heads] BF16 per block (compute)
constexpr uint32_t cb_scores = 17;  // scores, one BF16 tile per 32 blocks: row = position in block, col = block
constexpr uint32_t cb_send = 18;    // this core's fused survivor on its way to its partner (compute -> writer)
constexpr uint32_t cb_out = 19;     // a root's fused survivor on its way to DRAM (compute -> writer)

// Words of the meta page.
constexpr uint32_t meta_blocks = 0;  // 32-position blocks of this bank that hold a visible position
constexpr uint32_t meta_valid = 1;   // visible positions of this bank, a prefix of its stream
constexpr uint32_t meta_centre = 2;  // BF16 bits of minus the user's score centre, 0 when there is none

// A user's page of the step tensor (uint32, 128 bytes per user), read into the meta page; the reader then
// overwrites words 0..2 with the meta words above.
constexpr uint32_t step_page_bytes = 128;
constexpr uint32_t step_context = 4;   // positions the user sees
constexpr uint32_t step_centre = 5;    // BF16 bits of the user's score centre (scores leave as BF16(score - centre))
constexpr uint32_t step_weights = 16;  // words 16..31: the 32 head-sum multipliers as BF16 (signs when folded)

constexpr uint32_t key_tile_bytes = 1088;                 // BFP8: 64 exponent bytes + 1024 mantissa bytes
constexpr uint32_t key_block_tiles = 4;                   // 128 key dims
constexpr uint32_t key_block_bytes = 4 * key_tile_bytes;  // one 32-position block = one ND shard
constexpr uint32_t bf16_tile_bytes = 2048;
constexpr uint32_t word_tile_bytes = 4096;
constexpr uint32_t survivor_tiles = 2;  // 2048 fused [bf16 score | u16 stamp] words
constexpr uint32_t topk = 2048;

// BF16 bit patterns.
constexpr uint16_t bf16_one = 0x3F80;
constexpr uint16_t bf16_neg = 0xF180;  // -(2^100): the reference's NEG, exact in BF16 and FP32

// Word offset of element (row, col) of a 32x32 tile in its page (4 faces of 16x16, row-major inside a face).
constexpr uint32_t tile_word(uint32_t row, uint32_t col) {
    return ((((row >> 4) << 1) | (col >> 4)) << 8) | ((row & 15) << 4) | (col & 15);
}

// Program semaphores, as offsets from the id the program gives the piece.
constexpr uint32_t sem_credit = 0;  // receiver -> sender: the landing CB is free
constexpr uint32_t sem_data = 1;    // sender -> receiver: the survivor has landed

}  // namespace hy4_scan
