// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine, weight side of a distributed matmul (NCRISC). Grid column c computes output columns
// [c * nt, (c + 1) * nt); its top core streams the column's weights from DRAM, K block by K block
// ([kb, nt] tiles, stored contiguously per column), and multicasts each block down the column into every
// column core's cb_in1 (itself included). Receivers reserve cb_in1 first and count themselves ready on the
// top core.
//
// Compile-time args: 0 kb, 1 nt, 2 blocks, 3 R. Runtime args: 0 grid row r, 1 weights address (DRAM
// interleaved, pages (c, k, n)), 2 grid column c, 3 column rectangle noc x0, 4 y0, 5 x1, 6 y1,
// 7 top core noc x, 8 y.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in1 = 1;
constexpr uint32_t sem_ready = 2, sem_valid = 3;

void kernel_main() {
    constexpr uint32_t kb = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t R = get_compile_time_arg_val(3);
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);
    const uint32_t x0 = get_arg_val<uint32_t>(3), y0 = get_arg_val<uint32_t>(4);
    const uint32_t x1 = get_arg_val<uint32_t>(5), y1 = get_arg_val<uint32_t>(6);
    const uint32_t tx = get_arg_val<uint32_t>(7), ty = get_arg_val<uint32_t>(8);
    constexpr uint32_t tile = get_tile_size(cb_in1);
    constexpr uint32_t block_tiles = kb * nt;
    const InterleavedAddrGenFast<true> w{
        .bank_base_address = w_addr, .page_size = tile, .data_format = get_dataformat(cb_in1)};

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));
    const uint32_t valid_addr = get_semaphore(sem_valid);
    const uint32_t ready_addr = get_semaphore(sem_ready);
    uint32_t page = c * blocks * block_tiles;

    for (uint32_t b = 0; b < blocks; b++) {
        cb_reserve_back(cb_in1, block_tiles);
        const uint32_t dst = get_write_ptr(cb_in1);
        if (r == 0) {
            for (uint32_t t = 0; t < block_tiles; t++) {
                noc_async_read_tile(page + t, w, dst + t * tile);
            }
            page += block_tiles;
            noc_async_read_barrier();
            if constexpr (R > 1) {
                noc_semaphore_wait(ready, R - 1);
                noc_semaphore_set(ready, 0);
                noc_async_write_multicast(dst, get_noc_multicast_addr(x0, y0, x1, y1, dst), block_tiles * tile, R - 1);
                noc_async_write_barrier();
                *valid = 1;
                noc_semaphore_set_multicast(valid_addr, get_noc_multicast_addr(x0, y0, x1, y1, valid_addr), R - 1);
                noc_async_write_barrier();
            }
        } else {
            noc_semaphore_inc(get_noc_addr(tx, ty, ready_addr), 1);
            noc_semaphore_wait(valid, 1);
            noc_semaphore_set(valid, 0);
        }
        cb_push_back(cb_in1, block_tiles);
    }
    noc_async_atomic_barrier();
}
