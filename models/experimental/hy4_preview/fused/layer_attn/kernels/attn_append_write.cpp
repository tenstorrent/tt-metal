// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The append core's row writer in program P1 of the attention half (fused/layer_attn/program.py; PLAN.md section
// 17). The compute kernel leaves, for the 8 token rows: k_pass as scaled e4m3 (4 blocks of 128 values, row-major),
// the 4 block scales (Float32, column 0 of a tile each), k_rope after the rotation (2 BF16 tiles) and the BF16
// values of k_pass the cast saw (16 tiles). For every token row whose position this chip owns,
//     chip = (P / 128) % 32,   stripe row = (P / 4096) * 128 + P % 128,
// and whose frame the row's table maps, this kernel assembles the store's 656-byte row
//     [512 e4m3 | 4 Float32 scales | 64 BF16 rope values]
// and writes it where fused/kv_store's frame table puts that stripe row (kv_frames::locate). The write is drained
// before the kernel ends: P2's row readers start after P1 has ended on this chip.
// A position of 0xFFFFFFFF (or anything at or beyond 2^27) is an idle token row: nothing is written for it.
// With a keep address the BF16 rows (k_pass | k_rope, 18 small tiles) are also left in this core's arena for tests.
//
// Compile-time args: cb_e4m3, cb_scale, cb_rope, cb_keep, cb_rows (8 pages of 704 B, never pushed), cb_pos (one
// page of 64 B, never pushed), cb_frames (the frame tables: an L1 tensor, 128 words per token row).
// Runtime args: this chip's index in the stripe, the position tensor's address (uint32, DRAM, page 0), the store's
// address, the layer's slot step in a bank (bytes), keep address or 0.
// Stamp (HY4L_TIMING): 4 = the rows are written (the core's last stamp word).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"
#include "kv_frames.hpp"

constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(0);
constexpr uint32_t cb_scale = get_compile_time_arg_val(1);
constexpr uint32_t cb_rope = get_compile_time_arg_val(2);
constexpr uint32_t cb_keep = get_compile_time_arg_val(3);
constexpr uint32_t cb_rows = get_compile_time_arg_val(4);
constexpr uint32_t cb_pos = get_compile_time_arg_val(5);
constexpr uint32_t cb_frames = get_compile_time_arg_val(6);

namespace {
constexpr uint32_t USERS = 8;
constexpr uint32_t PASS = 16, ROPE = 2, BLOCKS = 4;
constexpr uint32_t TINY = 512, TILE16 = 2048, TILE32 = 4096;
constexpr uint32_t STRIDE = 704;           // a row's DRAM page
constexpr uint32_t ROW_BYTES = 656;        // 512 e4m3 | 16 scales | 128 rope
constexpr uint32_t BLOCK_PAGE = 4 * 1024;  // one block of the cast's output: 32 rows x 128 bytes
}  // namespace

void kernel_main() {
    const uint32_t chip = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t kv_addr = get_arg_val<uint32_t>(2);
    const uint32_t layer_bytes = get_arg_val<uint32_t>(3);
    const uint32_t keep_addr = get_arg_val<uint32_t>(4);

    cb_reserve_back(cb_pos, 1);
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pos));
    noc_async_read(get_noc_addr_from_bank_id<true>(0, pos_addr), get_write_ptr(cb_pos), 64);
    noc_async_read_barrier();
    cb_reserve_back(cb_rows, USERS);
    const uint32_t rows = get_write_ptr(cb_rows);
    volatile tt_l1_ptr uint32_t* frames = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_frames));

    cb_wait_front(cb_keep, PASS);
    cb_wait_front(cb_rope, ROPE);
    cb_wait_front(cb_scale, BLOCKS);
    cb_wait_front(cb_e4m3, BLOCKS * 4);
    volatile tt_l1_ptr uint32_t* e4m3 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_e4m3));
    volatile tt_l1_ptr uint32_t* scale = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scale));
    volatile tt_l1_ptr uint32_t* rope = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_rope));

    for (uint32_t u = 0; u < USERS; ++u) {
        const uint32_t position = positions[u];
        if (position >= (1u << 27) || kv_frames::chip_of(position) != chip) {
            continue;
        }
        const uint32_t row = kv_frames::stripe_row(position);
        volatile tt_l1_ptr uint32_t* table = frames + u * kv_frames::TABLE_FRAMES;
        if (!kv_frames::mapped(table, row)) {
            continue;  // beyond the conversation's frames: the host reserves a frame before a pass may write it
        }
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows + u * STRIDE);
        for (uint32_t b = 0; b < BLOCKS; ++b) {  // row u of block b: 128 bytes
            for (uint32_t i = 0; i < 32; ++i) {
                out[b * 32 + i] = e4m3[(b * BLOCK_PAGE + u * 128) / 4 + i];
            }
            out[128 + b] = scale[b * (TILE32 / 4) + u * 16];  // (row u, column 0) of the block's scale tile
        }
        for (uint32_t t = 0; t < ROPE; ++t) {  // row u of rope tile t: columns 0..15 in face 0, 16..31 in face 1
            for (uint32_t h = 0; h < 2; ++h) {
                for (uint32_t i = 0; i < 8; ++i) {
                    out[132 + t * 16 + h * 8 + i] = rope[t * (TILE16 / 4) + h * (TILE16 / 16) + u * 8 + i];
                }
            }
        }
        const kv_frames::Loc at = kv_frames::locate(table, layer_bytes, row, STRIDE);
        noc_async_write(rows + u * STRIDE, get_noc_addr_from_bank_id<true>(at.bank, kv_addr + at.offset), ROW_BYTES);
    }
    if (keep_addr != 0) {
        // the BF16 rows as 18 small tiles: [columns 0..15 of the 8 rows | columns 16..31] per tile
        const uint32_t keep = get_read_ptr(cb_keep);
        const uint32_t roped = get_read_ptr(cb_rope);
        for (uint32_t t = 0; t < PASS + ROPE; ++t) {  // (copies inside this core's L1, done by the NoC)
            const uint32_t src = t < PASS ? keep + t * TILE16 : roped + (t - PASS) * TILE16;
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(src + h * (TILE16 / 4), get_noc_addr(keep_addr + t * TINY + h * (TINY / 2)), TINY / 2);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_keep, PASS);
    cb_pop_front(cb_rope, ROPE);
    cb_pop_front(cb_scale, BLOCKS);
    cb_pop_front(cb_e4m3, BLOCKS * 4);
    hy4l::stamp(4);
}
