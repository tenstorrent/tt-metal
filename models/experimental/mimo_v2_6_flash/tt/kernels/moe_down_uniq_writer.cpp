// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC (NOC 1): the distinct experts from the stick, their h tiny tiles from the row-major activation (page j = the
// stick's expert j's kt [rows, 32] tiles) by ascending expert id (the reader's w_down order), then the user rows
// 0..rows-1 (rows 4 or 8) of this core's output columns (tile column t = j * banks + bank) into the compact [1, 1, 32,
// hidden * rows / 32] output: rows rows * c .. rows * c + rows - 1 of compact tile t % compact_tiles, c = t /
// compact_tiles (the row all-reduce then moves only the user rows; moe_expand_rows.cpp puts them back into the user
// rows of the [32, hidden] result).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t act_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t groups = get_compile_time_arg_val(5);
    constexpr uint32_t ct = get_compile_time_arg_val(6);
    constexpr uint32_t banks = get_compile_time_arg_val(7);
    constexpr uint32_t tiny_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t compact_tiles = get_compile_time_arg_val(9);
    // > 0: index of runtime args [count, x0, y0, ...]: once the output is written, raise semaphore signal_sem of
    // each of those cores by one (the row all-gather's senders in the same launch wait for every bank core).
    constexpr uint32_t signal_args_at = get_compile_time_arg_val(10);
    constexpr uint32_t signal_sem = get_compile_time_arg_val(11);
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(13);
    constexpr uint32_t tiny_face_bytes = tiny_bytes / 2;  // the user rows of one face
    constexpr uint32_t blocks_per_face = mimo_moe::kBf16FaceBytes / tiny_face_bytes;
    constexpr uint32_t bf16_tile_bytes = 2048;

    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t ids_addr = get_arg_val<uint32_t>(3);

    // The experts (the reader keeps its own copy of the stick in cb_ids).
    const uint32_t ids_l1 = get_write_ptr(cb_ids) + ids_bytes;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, ids_addr), ids_l1, ids_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_l1);
    const uint32_t experts = ids[0];
    uint32_t order[num_pairs];
    mimo_moe::experts_by_id(ids, experts, order);

    const InterleavedAddrGen<false> act_gen = {.bank_base_address = act_addr, .page_size = act_page_bytes};
    cb_reserve_back(cb_h, experts * kt);
    const uint32_t h_l1 = get_write_ptr(cb_h);
    for (uint32_t p = 0; p < experts; p++) {
        noc_async_read(act_gen.get_noc_addr(order[p], 0), h_l1 + p * act_page_bytes, act_page_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_h, experts * kt);

    const InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = bf16_tile_bytes};
    for (uint32_t g = 0; g < groups; g++) {
        cb_wait_front(cb_out, (g + 1) * ct);
        const uint32_t base = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < ct; c++) {
            const uint32_t j = g * ct + c;
            const uint32_t t = j * banks + bank;
            const uint32_t blk = t / compact_tiles;
            const uint32_t page = t - blk * compact_tiles;
            const uint32_t off =
                (blk / blocks_per_face) * 2 * mimo_moe::kBf16FaceBytes + (blk % blocks_per_face) * tiny_face_bytes;
            const uint32_t src = base + j * tiny_bytes;
            noc_async_write(src, out_gen.get_noc_addr(page, off), tiny_face_bytes);
            noc_async_write(
                src + tiny_face_bytes, out_gen.get_noc_addr(page, off + mimo_moe::kBf16FaceBytes), tiny_face_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, groups * ct);
    if constexpr (signal_args_at > 0) {
        const uint32_t n = get_arg_val<uint32_t>(signal_args_at);
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t sx = get_arg_val<uint32_t>(signal_args_at + 1 + 2 * i);
            const uint32_t sy = get_arg_val<uint32_t>(signal_args_at + 2 + 2 * i);
            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(signal_sem)), 1);
        }
        noc_async_atomic_barrier();
    }
}
