// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The key core's row writer in program P1 of a full-indexer layer (fused/layer_attn/program.py; PLAN.md section
// 20). The compute kernel leaves the 8 token rows' index keys as rows 0..7 of 4 BFP8 tiles (and, as BF16, the keys
// before the Hadamard basis). For every token row whose position this chip owns,
//     chip = (P / 128) % 32,   stripe row = (P / 4096) * 128 + P % 128,
// and whose frame the row's table maps, the key goes into the layer's key store: the 32-row block that holds the
// stripe row (kv_frames::locate with row stride 0: 4 tiles, 4,352 B, in one DRAM bank) is read, the row's 2 exponent
// bytes and 2 x 16 mantissa bytes per tile are replaced, the block is written back.
// The rows of a pass are patched BLOCK BY BLOCK (attn_key_rows.hpp): one read of every block that gets a row (issued
// at the kernel's start, while the compute kernel works), every row of the block patched into that copy, one
// write; the writes are drained before the kernel ends: the scan is this chip's next program. Two token rows at the
// same stripe row of one block: status 0xDE (below), the higher token row's key is stored.
// A position of 0xFFFFFFFF (or anything at or beyond 2^27) is an idle token row: nothing is read or written for it.
// With a keep address (tests) this core's arena also gets, from that address on: the keys before the Hadamard
// basis (4 small BF16 tiles, 2,048 B) | the packed tiles (4,352 B) | 16 words: the number of groups, then where[8]
// (the block of each token row as bank | offset, 0xFFFFFFFF: not written here) | per group, 4,352 B each, 8 slots:
// the block as it was READ | per group, 8 slots: the block as it is in DRAM after the pass (read again after the
// writes have drained).
// HY4A_KEY_DRY (bring-up: the layout check before anything is patched): the blocks are read and kept, nothing is
// patched or written.
//
// Compile-time args: cb_bfp, cb_keep, cb_blocks (8 pages of 4,352 B, never pushed), cb_pos (one page of 64 B, never
// pushed), cb_frames (the key store's frame tables: an L1 tensor, 128 words per token row).
// Runtime args: this chip's index in the stripe, the position tensor's address (uint32, DRAM, page 0), the key
// store's address, the layer's slot step in a bank (bytes), keep address or 0.
// Stamp (HY4L_TIMING): 4 = the blocks are written (the core's last stamp word; 0 to 3 are the feeder's).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"
#include "attn_key_rows.hpp"
#include "kv_frames.hpp"

constexpr uint32_t cb_bfp = get_compile_time_arg_val(0);
constexpr uint32_t cb_keep = get_compile_time_arg_val(1);
constexpr uint32_t cb_blocks = get_compile_time_arg_val(2);
constexpr uint32_t cb_pos = get_compile_time_arg_val(3);
constexpr uint32_t cb_frames = get_compile_time_arg_val(4);

namespace {
constexpr uint32_t USERS = 8;
constexpr uint32_t TINY = 512, TILE16 = 2048;
// this core's status word (attn_common.hpp A_STATUS): two token rows of the pass go to the same stripe row of one
// block; low bits: the two token rows, u | v << 8. The pass completes: the higher token row's key is the one stored.
constexpr uint32_t ST_KEY_SAME_ROW = 0xDEu << 24;
}  // namespace

void kernel_main() {
    const uint32_t chip = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t key_addr = get_arg_val<uint32_t>(2);
    const uint32_t layer_bytes = get_arg_val<uint32_t>(3);
    const uint32_t keep_addr = get_arg_val<uint32_t>(4);

    cb_reserve_back(cb_pos, 1);
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pos));
    noc_async_read(get_noc_addr_from_bank_id<true>(0, pos_addr), get_write_ptr(cb_pos), 64);
    noc_async_read_barrier();
    cb_reserve_back(cb_blocks, USERS);
    const uint32_t blocks = get_write_ptr(cb_blocks);
    volatile tt_l1_ptr uint32_t* frames = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames));

    // which block each token row goes to (bank and offset as one number: an offset is a multiple of 16), the groups
    uint32_t where[USERS], row_of[USERS], group[USERS], leader[USERS];
    for (uint32_t u = 0; u < USERS; ++u) {
        where[u] = key_rows::NONE;
        row_of[u] = 0;
        const uint32_t position = positions[u];
        if (position >= (1u << 27) || kv_frames::chip_of(position) != chip) {
            continue;
        }
        const uint32_t row = kv_frames::stripe_row(position);
        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;
        if (!kv_frames::mapped(table, row)) {
            continue;  // beyond the conversation's frames: the host reserves a frame before a pass may write it
        }
        const kv_frames::Loc at = kv_frames::locate(table, layer_bytes, row, 0);
        where[u] = at.offset | at.bank;
        row_of[u] = row % key_rows::BLOCK_ROWS;
    }
    const uint32_t groups = key_rows::group_rows(where, group, leader);
    const uint32_t twice = key_rows::same_row(where, row_of);
    if (twice != key_rows::NONE) {  // one position given to two token rows of a conversation: reported, not refused
        hy4a::fail(ST_KEY_SAME_ROW, twice);
    }
    auto block_addr = [&](uint32_t g) {
        const uint32_t w = where[leader[g]];
        return get_noc_addr_from_bank_id<true>(w & 15u, key_addr + (w & ~15u));
    };
    for (uint32_t g = 0; g < groups; ++g) {  // ONE read per block
        noc_async_read(block_addr(g), blocks + g * key_rows::BLOCK_BYTES, key_rows::BLOCK_BYTES);
    }
    noc_async_read_barrier();

    constexpr uint32_t K_ROT = 0, K_PACKED = key_rows::TILES * TINY, K_INFO = K_PACKED + key_rows::BLOCK_BYTES;
    constexpr uint32_t K_BEFORE = K_INFO + 64, K_AFTER = K_BEFORE + USERS * key_rows::BLOCK_BYTES;
    if (keep_addr != 0) {
        volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(keep_addr + K_INFO);
        info[0] = groups;
        for (uint32_t u = 0; u < USERS; ++u) {
            info[1 + u] = where[u];
        }
        for (uint32_t g = 0; g < groups; ++g) {  // (copies inside this core's L1, done by the NoC)
            noc_async_write(
                blocks + g * key_rows::BLOCK_BYTES,
                get_noc_addr(keep_addr + K_BEFORE + g * key_rows::BLOCK_BYTES),
                key_rows::BLOCK_BYTES);
        }
        noc_async_write_barrier();
    }

    cb_wait_front(cb_keep, key_rows::TILES);
    cb_wait_front(cb_bfp, key_rows::TILES);
    volatile tt_l1_ptr uint8_t* packed = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_read_ptr(cb_bfp));
#ifndef HY4A_KEY_DRY
    for (uint32_t u = 0; u < USERS; ++u) {  // token rows ascending: of two rows at one stripe row the higher stays
        if (group[u] != key_rows::NONE) {
            volatile tt_l1_ptr uint8_t* block =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(blocks + group[u] * key_rows::BLOCK_BYTES);
            key_rows::move_row(packed, u, block, row_of[u]);
        }
    }
    for (uint32_t g = 0; g < groups; ++g) {  // ONE write per block
        noc_async_write(blocks + g * key_rows::BLOCK_BYTES, block_addr(g), key_rows::BLOCK_BYTES);
    }
#endif
    if (keep_addr != 0) {
        // the keys before the Hadamard basis as 4 small tiles ([columns 0..15 of the 8 rows | columns 16..31] per
        // tile), then the packed tiles as they are
        const uint32_t keep = get_read_ptr(cb_keep);
        for (uint32_t t = 0; t < key_rows::TILES; ++t) {  // (copies inside this core's L1, done by the NoC)
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(
                    keep + t * TILE16 + h * (TILE16 / 4),
                    get_noc_addr(keep_addr + K_ROT + t * TINY + h * (TINY / 2)),
                    TINY / 2);
            }
        }
        noc_async_write(get_read_ptr(cb_bfp), get_noc_addr(keep_addr + K_PACKED), key_rows::BLOCK_BYTES);
    }
    noc_async_write_barrier();
    if (keep_addr != 0) {  // the blocks as DRAM holds them now
        for (uint32_t g = 0; g < groups; ++g) {
            noc_async_read(block_addr(g), keep_addr + K_AFTER + g * key_rows::BLOCK_BYTES, key_rows::BLOCK_BYTES);
        }
        noc_async_read_barrier();
    }
    cb_pop_front(cb_keep, key_rows::TILES);
    cb_pop_front(cb_bfp, key_rows::TILES);
    hy4l::stamp(4);
}
