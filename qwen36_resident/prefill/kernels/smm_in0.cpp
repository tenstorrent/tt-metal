// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Small-M streaming matmul, activation multicast (BRISC, NOC0 writes only), then the output. Every core
// needs all of x [M, K], one K block (Mt x kb tiles) at a time. Block b belongs to core b % P, whose NCRISC
// reads it from DRAM into cb_xstage and raises the local flag x_ready; this RISC multicasts it into every
// grid core's cb_in0 (itself included). A core announces itself ready for block b + 1 as soon as its slot
// is free; the two blocks in flight count on their own semaphore pair (b % 2). The flag goes through the
// data's command buffer and VC, so it lands after the block, from a constant-1 source. At the end the
// core's [Mt, nt_j] output tiles go to out [M, N] (DRAM interleaved).
//
// Compile-time args: 0 Mt, 1 kb, 2 blocks, 3 P, 4 Nt, 5 nt. Runtime args: 0 core j, 1 grid rectangle noc
// x0, 2 y0, 3 x1, 4 y1, 5 out address, 6 first output tile column, 7 output columns, then P x (x, y) NoC
// coordinates of the grid cores (row-major).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_in0 = 0, cb_xstage = 2, cb_out = 16;
constexpr uint32_t sem_ready0 = 0, sem_valid0 = 1, sem_ready1 = 2, sem_valid1 = 3, sem_one = 4, sem_x = 5;
constexpr uint32_t peers = 8;
}  // namespace

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t kb = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t nt = get_compile_time_arg_val(5);
    constexpr uint32_t blk = Mt * kb;
    constexpr uint32_t tile = get_tile_size(cb_in0), bytes = blk * tile;
    const uint32_t j = get_arg_val<uint32_t>(0);
    const uint32_t x0 = get_arg_val<uint32_t>(1), y0 = get_arg_val<uint32_t>(2);
    const uint32_t x1 = get_arg_val<uint32_t>(3), y1 = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5), n0 = get_arg_val<uint32_t>(6), ncols = get_arg_val<uint32_t>(7);
    auto peer = [&](uint32_t p, uint32_t a) {
        return get_noc_addr(get_arg_val<uint32_t>(peers + 2 * p), get_arg_val<uint32_t>(peers + 2 * p + 1), a);
    };
    auto mcast = [&](uint32_t a) { return get_noc_multicast_addr(x0, y0, x1, y1, a); };
    const uint32_t ready_addr[2] = {get_semaphore(sem_ready0), get_semaphore(sem_ready1)};
    const uint32_t valid_addr[2] = {get_semaphore(sem_valid0), get_semaphore(sem_valid1)};
    volatile tt_l1_ptr uint32_t* x_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
    const uint32_t base = get_write_ptr(cb_in0), stage = get_write_ptr(cb_xstage);

    cb_reserve_back(cb_in0, blk);
    uint32_t dst = base;
    noc_semaphore_inc(peer(0, ready_addr[0]), 1);
    for (uint32_t b = 0; b < blocks; b++) {
        const uint32_t p = b & 1;
        volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr[p]);
        if (b % P == j) {
            noc_semaphore_wait(x_ready, 1);
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr[p]), P);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr[p]), 0);
            noc_async_write_multicast_loopback_src(stage, mcast(dst), bytes, P);
            noc_async_writes_flushed();
            noc_async_write_multicast_loopback_src(get_semaphore(sem_one), mcast(valid_addr[p]), 4, P);
            noc_async_writes_flushed();
        }
        if (b + 1 < blocks) {
            cb_reserve_back(cb_in0, 2 * blk);
            noc_semaphore_inc(peer((b + 1) % P, ready_addr[p ^ 1]), 1);
        }
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
        cb_push_back(cb_in0, blk);
        dst = dst == base ? base + bytes : base;
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    const uint32_t out_tile = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> out{
        .bank_base_address = out_addr, .page_size = out_tile, .data_format = DataFormat::Float16_b};
    cb_wait_front(cb_out, Mt * nt);
    for (uint32_t m = 0; m < Mt; m++) {
        for (uint32_t n = 0; n < ncols; n++) {
            noc_async_write_tile(m * Nt + n0 + n, out, get_read_ptr(cb_out) + (m * nt + n) * out_tile);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, Mt * nt);
}
