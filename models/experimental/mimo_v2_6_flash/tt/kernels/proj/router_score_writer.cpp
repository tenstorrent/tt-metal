// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BRISC: the user rows 0..rows-1 (rows 4, 8 or 16) of this core's kq bf16 x tiles [k0, k0 + kq) into cb_x (one read per
// tile covers those rows of faces 0 and 1; every read in flight at once). A non-leader then sends the user rows of its
// partial logits tile (faces 0 and 1) to slot q - 1 of its leader's cb_gather and increments the leader's semaphore.
// The leader reads the bias tile, hands compute the group's partials once all have landed, and writes the user rows of
// the scores and biased tiles into page out_page of the two fp32 DRAM-interleaved outputs, whose other rows the
// caller keeps.
//
// Only the user rows of x carry users and an output row depends only on the same x row, so the unread rows of cb_x
// reach only output rows that are never written.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <bool dram>
FORCE_INLINE uint64_t page_noc_addr(uint32_t base, uint32_t page, uint32_t page_bytes) {
    constexpr uint32_t banks = dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    return get_noc_addr_from_bank_id<dram>(page % banks, base + (page / banks) * page_bytes);
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(2);
    constexpr bool x_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t kq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_part = get_compile_time_arg_val(5);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(6);
    constexpr uint32_t group = get_compile_time_arg_val(7);
    constexpr uint32_t sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t rows = get_compile_time_arg_val(9);  // the user rows: 4, 8 or 16
    constexpr uint32_t x_tile_bytes = 2048;                 // bf16 32x32
    constexpr uint32_t x_face_bytes = 512;                  // bf16 16x16
    constexpr uint32_t x_user_bytes = rows * 32;            // the user rows of a bf16 face
    constexpr uint32_t x_read_bytes = x_face_bytes + x_user_bytes;
    constexpr uint32_t out_tile_bytes = 4096;       // fp32 32x32
    constexpr uint32_t out_face_bytes = 1024;       // fp32 16x16
    constexpr uint32_t out_user_bytes = rows * 64;  // the user rows of an fp32 face

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);
    const uint32_t biased_addr = get_arg_val<uint32_t>(2);
    const uint32_t bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_page = get_arg_val<uint32_t>(4);
    const uint32_t q = get_arg_val<uint32_t>(5);
    const uint32_t k0 = get_arg_val<uint32_t>(6);
    const uint32_t leader_x = get_arg_val<uint32_t>(7);
    const uint32_t leader_y = get_arg_val<uint32_t>(8);

    if (q == 0) {
        cb_reserve_back(cb_bias, 1);
        noc_async_read(
            page_noc_addr<true>(bias_addr, out_page, out_tile_bytes), get_write_ptr(cb_bias), out_tile_bytes);
    }
    cb_reserve_back(cb_x, kq);
    uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t k = k0; k < k0 + kq; k++) {
        noc_async_read(page_noc_addr<x_dram>(x_addr, k, x_tile_bytes), x_l1, x_read_bytes);
        x_l1 += x_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kq);

    const uint32_t sem_addr = get_semaphore(sem_id);
    if (q != 0) {
        cb_wait_front(cb_part, 1);
        const uint32_t src = get_read_ptr(cb_part);
        const uint32_t slot = get_write_ptr(cb_gather) + (q - 1) * out_tile_bytes;
        noc_async_write(src, get_noc_addr(leader_x, leader_y, slot), out_user_bytes);
        noc_async_write(src + out_face_bytes, get_noc_addr(leader_x, leader_y, slot + out_face_bytes), out_user_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, sem_addr), 1);
        cb_pop_front(cb_part, 1);
        noc_async_atomic_barrier();
        return;
    }

    cb_push_back(cb_bias, 1);
    if constexpr (group > 1) {
        cb_reserve_back(cb_gather, group - 1);
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        noc_semaphore_wait(sem, group - 1);
        noc_semaphore_set(sem, 0);
        cb_push_back(cb_gather, group - 1);
    }
    cb_wait_front(cb_out, 2);
    const uint32_t src = get_read_ptr(cb_out);
    const uint32_t addrs[2] = {scores_addr, biased_addr};
    for (uint32_t i = 0; i < 2; i++) {
        const uint64_t dst = page_noc_addr<true>(addrs[i], out_page, out_tile_bytes);
        const uint32_t s = src + i * out_tile_bytes;
        noc_async_write(s, dst, out_user_bytes);
        noc_async_write(s + out_face_bytes, dst + out_face_bytes, out_user_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 2);
}
