// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1) of the deduplicated MoE stream with gate|up and down in one launch (tt/moe_stream.py FusedStream).
//
// Gate|up side: x rows 0..rows-1 as [rows, 32] tiny tiles; once every bank core's reader has filled the routing
// table (route_sem counts banks; see moe_fused_uniq_reader.cpp), the distinct-expert list, its count for compute
// (cb_cnt) and one routing-weight tile per distinct expert (cb_wt).
//
// Then the h exchange: this core's h tile of distinct expert j (h column `bank`) goes into slot j * kt + bank of
// cb_hall on every bank core (the CB sits at the same L1 address on all of them), so each core ends up with every
// expert's whole [rows, 256] h for the down matmul. Semaphore `ready` (every core raises it on every core once its
// cb_hall is reserved, before any h lands there) and `done` (raised once all of this core's h writes have landed) order
// it.
//
// Down side as moe_down_uniq_writer.cpp: the user rows of this core's output columns (tile column t = c * banks + bank)
// into the compact [1, 1, 32, hidden * rows / 32] output, then optionally a semaphore raise on each signal core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"
#include "moe_route_select.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(3);
    constexpr uint32_t topk = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tbl = get_compile_time_arg_val(5);
    constexpr uint32_t users = get_compile_time_arg_val(6);
    constexpr uint32_t route_sem = get_compile_time_arg_val(7);
    constexpr uint32_t entry_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(9);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(10);
    constexpr uint32_t rows = get_compile_time_arg_val(11);  // the tiny tiles' rows: users <= rows
    constexpr uint32_t cb_hall = get_compile_time_arg_val(12);
    constexpr uint32_t h_kt = get_compile_time_arg_val(13);  // h tiles per expert: one per bank core
    constexpr uint32_t banks = get_compile_time_arg_val(14);
    constexpr uint32_t ready_sem = get_compile_time_arg_val(15);
    constexpr uint32_t done_sem = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);
    constexpr uint32_t cols = get_compile_time_arg_val(18);  // output tiles per core
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(19);
    // > 0: index of runtime args [count, x0, y0, ...]: once the output is written, raise semaphore signal_sem of
    // each of those cores by one (the row all-gather's senders in the same launch wait for every bank core).
    constexpr uint32_t signal_args_at = get_compile_time_arg_val(20);
    constexpr uint32_t signal_sem = get_compile_time_arg_val(21);
    constexpr uint32_t experts = get_compile_time_arg_val(22);
    constexpr uint32_t bf16_tile_bytes = 2048;
    constexpr uint32_t tiny_face_bytes = rows * mimo_moe::kFaceDim * 2;  // rows 0..rows-1 of one face
    constexpr uint32_t tiny_bytes = 2 * tiny_face_bytes;
    constexpr uint32_t row_bytes = mimo_moe::kFaceDim * 2;  // one face row: 32 B
    constexpr uint32_t blocks_per_face = mimo_moe::kBf16FaceBytes / tiny_face_bytes;
    static_assert(users <= rows && rows <= mimo_moe::kFaceDim);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    constexpr uint32_t peers_at = 3;  // [x0, y0, ..., x_{banks-1}, y_{banks-1}]: the bank cores, bank order

    MOE_MARK(20);
    const InterleavedAddrGen<true> x_gen = {.bank_base_address = x_addr, .page_size = bf16_tile_bytes};
    cb_reserve_back(cb_x, kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kt; t++) {
        const uint32_t dst = x_l1 + t * 2 * tiny_face_bytes;
        noc_async_read(x_gen.get_noc_addr(t, 0), dst, tiny_face_bytes);
        noc_async_read(x_gen.get_noc_addr(t, mimo_moe::kBf16FaceBytes), dst + tiny_face_bytes, tiny_face_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);
    MOE_MARK(22);

    // Every user's selection, from the routing table.
    const uint32_t tbl = get_write_ptr(cb_tbl);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(route_sem)), banks);
    MOE_MARK(24);
    uint32_t sel_ids[users][topk];
    uint32_t sel_w[users][topk];
    bool active[users];
    for (uint32_t u = 0; u < users; u++) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl + u * entry_bytes);
        active[u] = e[0] != 0;
        for (uint32_t i = 0; i < topk; i++) {
            sel_ids[u][i] = e[1 + i];
            sel_w[u][i] = e[1 + topk + i];
        }
    }
    uint32_t uniq[num_pairs];
    const uint32_t n = mimo_moe::table_experts<topk, users, experts>(tbl, entry_bytes, uniq);
    cb_reserve_back(cb_cnt, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt)) = n;
    cb_push_back(cb_cnt, 1);
    MOE_MARK(25);

    // This core can take the other cores' h now: its cb_hall is reserved (and no earlier use of that L1 is pending).
    cb_reserve_back(cb_hall, n * h_kt);
    const uint32_t hall = get_write_ptr(cb_hall);  // the same address on every bank core
    const uint32_t ready_addr = get_semaphore(ready_sem);
    const uint32_t done_addr = get_semaphore(done_sem);
    for (uint32_t c = 0; c < banks; c++) {
        const uint32_t x = get_arg_val<uint32_t>(peers_at + 2 * c);
        const uint32_t y = get_arg_val<uint32_t>(peers_at + 2 * c + 1);
        noc_semaphore_inc(get_noc_addr(x, y, ready_addr), 1);
    }

    // Weight tiles (bf16 32 x 32; only rows 0..rows-1 are packed, the rest is never read): row u of faces 0 and 1.
    for (uint32_t j = 0; j < n; j++) {
        cb_reserve_back(cb_wt, 1);
        const uint32_t wt = get_write_ptr(cb_wt);
        for (uint32_t u = 0; u < rows; u++) {
            uint32_t bits = 0;
            if (u < users && active[u]) {
                for (uint32_t i = 0; i < topk; i++) {
                    if (sel_ids[u][i] == uniq[j]) {
                        bits = sel_w[u][i] >> 16;
                    }
                }
            }
            const uint32_t pair = (bits << 16) | bits;
            volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wt + u * row_bytes);
            volatile tt_l1_ptr uint32_t* f1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wt + mimo_moe::kBf16FaceBytes + u * row_bytes);
            for (uint32_t c = 0; c < mimo_moe::kFaceDim / 2; c++) {
                f0[c] = pair;
                f1[c] = pair;
            }
        }
        cb_push_back(cb_wt, 1);
    }
    MOE_MARK(26);

    // Every core's h tile of each expert, into every core's cb_hall.
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    noc_semaphore_wait(ready, banks);
    for (uint32_t j = 0; j < n; j++) {
        cb_wait_front(cb_h, 1);
        const uint32_t src = get_read_ptr(cb_h);
        const uint32_t dst = hall + (j * h_kt + bank) * tiny_bytes;
        for (uint32_t c = 0; c < banks; c++) {
            const uint32_t x = get_arg_val<uint32_t>(peers_at + 2 * c);
            const uint32_t y = get_arg_val<uint32_t>(peers_at + 2 * c + 1);
            noc_async_write(src, get_noc_addr(x, y, dst), tiny_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_h, 1);
        if (j == 0) {
            MOE_MARK(27);
        }
    }
    noc_async_write_barrier();
    for (uint32_t c = 0; c < banks; c++) {
        const uint32_t x = get_arg_val<uint32_t>(peers_at + 2 * c);
        const uint32_t y = get_arg_val<uint32_t>(peers_at + 2 * c + 1);
        noc_semaphore_inc(get_noc_addr(x, y, done_addr), 1);
    }
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_addr);
    noc_semaphore_wait(done, banks);
    cb_push_back(cb_hall, n * h_kt);
    MOE_MARK(28);

    // Down outputs: tile column t = c * banks + bank goes to rows rows * blk .. of compact tile t % compact_tiles,
    // blk = t / compact_tiles.
    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = bf16_tile_bytes};
    cb_wait_front(cb_out, cols);
    const uint32_t base = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; c++) {
        const uint32_t t = c * banks + bank;
        const uint32_t blk = t / compact_tiles;
        const uint32_t page = t - blk * compact_tiles;
        const uint32_t off =
            (blk / blocks_per_face) * 2 * mimo_moe::kBf16FaceBytes + (blk % blocks_per_face) * tiny_face_bytes;
        const uint32_t src = base + c * tiny_bytes;
        noc_async_write(src, out_gen.get_noc_addr(page, off), tiny_face_bytes);
        noc_async_write(
            src + tiny_face_bytes, out_gen.get_noc_addr(page, off + mimo_moe::kBf16FaceBytes), tiny_face_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
    // (ready and done start every launch at their initial 0: semaphores are part of the launch's kernel config.)
    if constexpr (signal_args_at > 0) {
        const uint32_t count = get_arg_val<uint32_t>(signal_args_at);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t sx = get_arg_val<uint32_t>(signal_args_at + 1 + 2 * i);
            const uint32_t sy = get_arg_val<uint32_t>(signal_args_at + 2 + 2 * i);
            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(signal_sem)), 1);
        }
        noc_async_atomic_barrier();
    }
    MOE_MARK(29);
}
