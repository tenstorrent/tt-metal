// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Residual add + RMSNorm, writer (BRISC): writes the reduce scaler; sends this core's part of the row sums
// to slot q of every row core's cb_recv (itself included) and counts the row's Q parts on the semaphore;
// stores s to x (in place) as it is produced, and h at the end.
// Compile-time args: 0 Ht, 1 W, 2 Q. Runtime args: 0 r, 1 c0, 2 q, 3 x address, 4 h address, then Q x (x, y)
// NoC coordinates of the row's cores.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t Q = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = 2, cb_part = 3, cb_recv = 4, cb_s = 16, cb_h = 17, sem = 0;
    constexpr uint32_t tile = get_tile_size(cb_s), part_tile = get_tile_size(cb_part);
    const uint32_t r = get_arg_val<uint32_t>(0), c0 = get_arg_val<uint32_t>(1), q = get_arg_val<uint32_t>(2);
    const InterleavedAddrGenFast<true> x{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> h{
        .bank_base_address = get_arg_val<uint32_t>(4), .page_size = tile, .data_format = DataFormat::Float16_b};

    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t w = 0; w < 512; w++) {
        sc[w] = (w % 128) < 8 ? 0x3f803f80 : 0;  // bf16 1.0 in the first row of each face
    }
    cb_push_back(cb_scaler, 1);

    // s to x as the compute produces it (cb_s stays for the compute's h)
    for (uint32_t i = 0; i < W; i += 2) {
        cb_wait_front(cb_s, i + 2);
        for (uint32_t t = i; t < i + 2; t++) {
            noc_async_write_tile(r * Ht + c0 + t, x, get_read_ptr(cb_s) + t * tile);
        }
    }

    const uint32_t recv = get_write_ptr(cb_recv), sem_addr = get_semaphore(sem);
    cb_wait_front(cb_part, 1);
    for (uint32_t p = 0; p < Q; p++) {
        noc_async_write(
            get_read_ptr(cb_part),
            get_noc_addr(get_arg_val<uint32_t>(5 + 2 * p), get_arg_val<uint32_t>(6 + 2 * p), recv + q * part_tile),
            part_tile);
    }
    noc_async_write_barrier();
    for (uint32_t p = 0; p < Q; p++) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(5 + 2 * p), get_arg_val<uint32_t>(6 + 2 * p), sem_addr), 1);
    }
    cb_pop_front(cb_part, 1);
    cb_reserve_back(cb_recv, Q);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), Q);
    cb_push_back(cb_recv, Q);

    for (uint32_t i = 0; i < W; i += 2) {
        cb_wait_front(cb_h, i + 2);
        for (uint32_t t = i; t < i + 2; t++) {
            noc_async_write_tile(r * Ht + c0 + t, h, get_read_ptr(cb_h) + t * tile);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_h, W);
    cb_pop_front(cb_s, W);
}
