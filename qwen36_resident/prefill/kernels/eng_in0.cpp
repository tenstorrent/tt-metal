// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine, activation side of a distributed matmul (BRISC). The cores form an R x Q grid; an
// activation [C, K] lives as (row group r, K column block c) on core (r, c): mt row tiles by this core's
// K blocks of kb tiles (row-major, resident in cb_act). The matmul walks the K blocks in order; block b
// belongs to the core column owning it, which multicasts it along its grid row into every row core's
// cb_in0 (itself included). Receivers reserve cb_in0 first and count themselves ready on the sender.
//
// Compile-time args: 0 mt, 1 kb, 2 blocks (K blocks of the matmul), 3 Q, 4 cb_act row width (tiles).
// Runtime args: 0 grid column c, 1 first block owned, 2 blocks owned, 3 row rectangle noc x0, 4 y0,
// 5 x1, 6 y1, 7 output address (DRAM interleaved; 0 = none), 8 output width (tiles), 9 grid row r,
// 10 nt, then Q x (x, y) NoC coordinates of the row's cores, then the owner column of each block.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in0 = 0, cb_act = 10;
constexpr uint32_t sem_ready = 0, sem_valid = 1;

void kernel_main() {
    constexpr uint32_t mt = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t Q = get_compile_time_arg_val(3);
    constexpr uint32_t act_row = get_compile_time_arg_val(4);
    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t owned = get_arg_val<uint32_t>(2);
    const uint32_t x0 = get_arg_val<uint32_t>(3), y0 = get_arg_val<uint32_t>(4);
    const uint32_t x1 = get_arg_val<uint32_t>(5), y1 = get_arg_val<uint32_t>(6);
    const uint32_t out_addr = get_arg_val<uint32_t>(7), out_w = get_arg_val<uint32_t>(8);
    const uint32_t r = get_arg_val<uint32_t>(9), nt = get_arg_val<uint32_t>(10);
    constexpr uint32_t peers = 11, owners = 11 + 2 * Q;
    constexpr uint32_t tile = get_tile_size(cb_in0);
    constexpr uint32_t block_tiles = mt * kb;

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));
    const uint32_t valid_addr = get_semaphore(sem_valid);
    const uint32_t ready_addr = get_semaphore(sem_ready);
    const uint32_t act = get_read_ptr(cb_act);

    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t owner = get_arg_val<uint32_t>(owners + b);
        cb_reserve_back(cb_in0, block_tiles);
        const uint32_t dst = get_write_ptr(cb_in0);
        if (owner == c) {
            noc_semaphore_wait(ready, Q - 1);
            noc_semaphore_set(ready, 0);
            const uint32_t k0 = (b - first) * kb;
            for (uint32_t i = 0; i < mt; i++) {
                noc_async_write_multicast_loopback_src(
                    act + (i * act_row + k0) * tile,
                    get_noc_multicast_addr(x0, y0, x1, y1, dst + i * kb * tile),
                    kb * tile,
                    Q);
            }
            noc_async_write_barrier();
            *valid = 1;
            noc_semaphore_set_multicast_loopback_src(valid_addr, get_noc_multicast_addr(x0, y0, x1, y1, valid_addr), Q);
            noc_async_write_barrier();
        } else {
            const uint32_t sx = get_arg_val<uint32_t>(peers + 2 * owner);
            const uint32_t sy = get_arg_val<uint32_t>(peers + 2 * owner + 1);
            noc_semaphore_inc(get_noc_addr(sx, sy, ready_addr), 1);
        }
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
        cb_push_back(cb_in0, block_tiles);
    }
    noc_async_atomic_barrier();

    if (out_addr) {
        constexpr uint32_t cb_acc = 16;
        const uint32_t out_tile = get_tile_size(cb_acc);
        const InterleavedAddrGenFast<true> out{
            .bank_base_address = out_addr, .page_size = out_tile, .data_format = get_dataformat(cb_acc)};
        cb_wait_front(cb_acc, mt * nt);
        for (uint32_t i = 0; i < mt; i++) {
            for (uint32_t j = 0; j < nt; j++) {
                noc_async_write_tile(
                    (r * mt + i) * out_w + c * nt + j, out, get_read_ptr(cb_acc) + (i * nt + j) * out_tile);
            }
        }
        noc_async_write_barrier();
    }
}
