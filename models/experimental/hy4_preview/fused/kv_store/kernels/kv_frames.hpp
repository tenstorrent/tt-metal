// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Where a stripe row of a PAGED conversation is in DRAM (fused/kv_store/frames.py is the same arithmetic on the
// host and states the layout). Free of host-only and kernel-only dependencies.
//
// A conversation is a list of frames of 8192 tokens: 256 stripe rows per chip = one 32-row shard in each of the 8
// DRAM banks, for every layer. Its TABLE (uint32, TABLE_FRAMES entries, resident in L1) holds, for the frame of
// stripe rows [256 k, 256 k + 256), the byte offset of that frame's layer-0 shard inside a bank ("frame base").
// The same table serves every layer, bank and chip; a kernel adds its layer's slot offset once per launch:
//
//     slot_bytes = layer * SLOT_STRIDE_SHARDS * SHARD_BYTES     (1 shard for a frame-major pool; the slot's
//                                                               capacity in frames for a slot-major cache)
//     Loc at = kv_frames::locate(table, slot_bytes, row, ROW_STRIDE);
//     noc address = get_noc_addr_from_bank_id<true>(at.bank, buffer_address + at.offset)
//
// ROW_STRIDE = 704 gives the packed latent row; ROW_STRIDE = 0 the 32-row block that holds the row (index keys:
// 4 BFP8 tiles, 4352 bytes). Cost per call: one L1 load, two shifts, two masks, two adds and a multiply by a
// constant (none with ROW_STRIDE 0).
#pragma once

#include <cstdint>

namespace kv_frames {

constexpr uint32_t STRIPE_BLOCK_LOG2 = 7;  // 128 consecutive positions per chip and turn
constexpr uint32_t CHIPS_LOG2 = 5;         // 32 chips, linear chip = 4 * mesh row + mesh column
constexpr uint32_t SHARD_ROWS_LOG2 = 5;    // 32 rows per ND shard
constexpr uint32_t BANKS_LOG2 = 3;         // 8 DRAM banks
constexpr uint32_t FRAME_ROWS_LOG2 = SHARD_ROWS_LOG2 + BANKS_LOG2;    // 256 stripe rows per chip
constexpr uint32_t FRAME_TOKENS_LOG2 = FRAME_ROWS_LOG2 + CHIPS_LOG2;  // 8192
constexpr uint32_t TABLE_FRAMES = 128;  // entries of one decode row's table: 1,048,576 tokens
constexpr uint32_t TABLE_BYTES = TABLE_FRAMES * sizeof(uint32_t);
constexpr uint32_t UNMAPPED = 0xFFFFFFFFu;  // the conversation has no such frame

struct Loc {
    uint32_t bank;
    uint32_t offset;  // bytes from the tensor's buffer address, in that bank
};

// THE function. `table`: the frame bases of the conversation the decode row serves (any pointer to uint32).
template <typename Table>
inline __attribute__((always_inline)) Loc locate(Table table, uint32_t slot_bytes, uint32_t row, uint32_t row_stride) {
    return Loc{
        (row >> SHARD_ROWS_LOG2) & ((1u << BANKS_LOG2) - 1),
        table[row >> FRAME_ROWS_LOG2] + slot_bytes + (row & ((1u << SHARD_ROWS_LOG2) - 1)) * row_stride};
}

// Is the frame of stripe row `row` there? (An append beyond the frame list is skipped; rows < TABLE_FRAMES * 256.)
template <typename Table>
inline __attribute__((always_inline)) bool mapped(Table table, uint32_t row) {
    return (row >> FRAME_ROWS_LOG2) < TABLE_FRAMES && table[row >> FRAME_ROWS_LOG2] != UNMAPPED;
}

// The stripe: position -> chip and stripe row (tt/hy4/kv_cache.py KvStripe; no capacity in either).
constexpr uint32_t chip_of(uint32_t position) { return (position >> STRIPE_BLOCK_LOG2) & ((1u << CHIPS_LOG2) - 1); }
constexpr uint32_t stripe_row(uint32_t position) {
    return ((position >> (STRIPE_BLOCK_LOG2 + CHIPS_LOG2)) << STRIPE_BLOCK_LOG2) |
           (position & ((1u << STRIPE_BLOCK_LOG2) - 1));
}

// The stripe row at which the j-th 32-row block of `bank`'s share of a slot starts (a per-bank key stream reads
// block j of bank b: it is the bank's ONE shard of frame j).
constexpr uint32_t stream_row(uint32_t bank, uint32_t j) { return ((j << BANKS_LOG2) | bank) << SHARD_ROWS_LOG2; }

}  // namespace kv_frames
