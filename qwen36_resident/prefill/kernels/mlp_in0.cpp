// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine MLP stack, activation mover (BRISC, NOC0); see mlp_compute.cpp for the layout. Per layer:
//   norm: multicast this core's row sums of x^2 (mt tiles) into slot c of every row core's cb_recv, then
//         count the row's Q arrivals on sem_norm
//   G, U: K block b of h belongs to grid column b / gu_blocks, which multicasts it along the row into
//         every row core's cb_in0 (itself included); receivers reserve cb_in0 first and count themselves
//         ready on the sender
//   D:    the same walk over the blocks of a
// Every block lands in an mt x S0 cb_in0 page group (kb valid columns per row), so all cores' cb_in0
// write pointers move in step and a sender writes to its own write pointer on every receiver.
//
// Compile-time args: 0 mt, 1 W, 2 IC, 3 Q, 4 S0, 5 GU blocks per owner, 6 GU slot, 7 HC, 8 D blocks per
// owner, 9 D slot, 10 layers. Runtime args: 0 grid column c, 1 row rectangle noc x0, 2 y0, 3 x1, 4 y1,
// then Q x (x, y) NoC coordinates of the row's cores.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_in0 = 0, cb_h = 11, cb_a = 12, cb_part = 16, cb_recv = 17, cb_scaler = 19;
constexpr uint32_t cb_h_rdy = 24, cb_a_rdy = 25;
constexpr uint32_t sem_ready0 = 0, sem_valid0 = 1, sem_norm = 4, sem_ready1 = 5, sem_valid1 = 6;
// holds 1: the flag source, so the sender's own flag is set only by the multicast, after its data. The
// flag goes through the data's command buffer and VC, so it lands after the data.
constexpr uint32_t sem_one = 7;
uint32_t ring_base;  // cb_in0 holds two blocks

constexpr uint32_t mt = get_compile_time_arg_val(0);
constexpr uint32_t W = get_compile_time_arg_val(1);
constexpr uint32_t IC = get_compile_time_arg_val(2);
constexpr uint32_t Q = get_compile_time_arg_val(3);
constexpr uint32_t S0 = get_compile_time_arg_val(4);
constexpr uint32_t gu_blocks = get_compile_time_arg_val(5);
constexpr uint32_t gu_slot = get_compile_time_arg_val(6);
constexpr uint32_t HC = get_compile_time_arg_val(7);
constexpr uint32_t d_blocks = get_compile_time_arg_val(8);
constexpr uint32_t d_slot = get_compile_time_arg_val(9);
constexpr uint32_t layers = get_compile_time_arg_val(10);
constexpr uint32_t tile = get_tile_size(cb_in0);
constexpr uint32_t peers = 5;

struct Row {
    uint32_t c, x0, y0, x1, y1;
    uint64_t mcast(uint32_t addr) const { return get_noc_multicast_addr(x0, y0, x1, y1, addr); }
    uint64_t peer(uint32_t q, uint32_t addr) const {
        return get_noc_addr(get_arg_val<uint32_t>(peers + 2 * q), get_arg_val<uint32_t>(peers + 2 * q + 1), addr);
    }
};

// walk the K blocks of src (mt rows of `stride` tiles; each owner holds `per` tiles in blocks of `slot`).
// A core announces itself ready for block b + 1 as soon as its slot is free, before block b lands; the two
// blocks in flight count on their own semaphore pair (b % 2).
void pass(const Row& row, uint32_t src, uint32_t stride, uint32_t blocks_per_owner, uint32_t slot, uint32_t per) {
    constexpr uint32_t blk = mt * S0, bytes = blk * tile;
    const uint32_t ready_addr[2] = {get_semaphore(sem_ready0), get_semaphore(sem_ready1)};
    const uint32_t valid_addr[2] = {get_semaphore(sem_valid0), get_semaphore(sem_valid1)};
    const uint32_t n = Q * blocks_per_owner;
    cb_reserve_back(cb_in0, blk);
    uint32_t dst = get_write_ptr(cb_in0);
    noc_semaphore_inc(row.peer(0, ready_addr[0]), 1);
    for (uint32_t b = 0; b < n; b++) {
        const uint32_t p = b & 1, owner = b / blocks_per_owner, i = b % blocks_per_owner;
        const uint32_t kb = i + 1 < blocks_per_owner ? slot : per - slot * (blocks_per_owner - 1);
        volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr[p]);
        if (owner == row.c) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr[p]), Q);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr[p]), 0);
            for (uint32_t m = 0; m < mt; m++) {
                noc_async_write_multicast_loopback_src(
                    src + (m * stride + i * slot) * tile, row.mcast(dst + m * S0 * tile), kb * tile, Q);
            }
            noc_async_writes_flushed();
            noc_async_write_multicast_loopback_src(get_semaphore(sem_one), row.mcast(valid_addr[p]), 4, Q);
            noc_async_writes_flushed();
        }
        if (b + 1 < n) {
            cb_reserve_back(cb_in0, 2 * blk);
            noc_semaphore_inc(row.peer((b + 1) / blocks_per_owner, ready_addr[p ^ 1]), 1);
        }
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
        cb_push_back(cb_in0, blk);
        dst = dst == ring_base ? ring_base + bytes : ring_base;
    }
    noc_async_write_barrier();
}
}  // namespace

void kernel_main() {
    const Row row{
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4)};
    volatile tt_l1_ptr uint32_t* norm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_norm));
    const uint32_t norm_addr = get_semaphore(sem_norm);
    const uint32_t recv = get_write_ptr(cb_recv);
    const uint32_t h = get_read_ptr(cb_h), a = get_read_ptr(cb_a);
    ring_base = get_write_ptr(cb_in0);

    // reduce scaler: bf16 1.0 in the first row of each face, written by stores (NCRISC owns this core's
    // NOC0 read state for the weight stream)
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t w = 0; w < 512; w++) {
        s[w] = (w % 128) < 8 ? 0x3f803f80 : 0;
    }
    cb_push_back(cb_scaler, 1);
    for (uint32_t l = 0; l < layers; l++) {
        cb_wait_front(cb_part, mt);
        noc_async_write_multicast_loopback_src(
            get_read_ptr(cb_part), row.mcast(recv + row.c * mt * tile), mt * tile, Q);
        noc_async_write_barrier();
        cb_pop_front(cb_part, mt);
        for (uint32_t q = 0; q < Q; q++) {
            noc_semaphore_inc(row.peer(q, norm_addr), 1);
        }
        cb_reserve_back(cb_recv, Q * mt);
        noc_semaphore_wait(norm, Q);
        noc_semaphore_set(norm, 0);
        cb_push_back(cb_recv, Q * mt);

        cb_wait_front(cb_h_rdy, 1);
        cb_pop_front(cb_h_rdy, 1);
        pass(row, h, W, gu_blocks, gu_slot, HC);
        pass(row, h, W, gu_blocks, gu_slot, HC);
        cb_wait_front(cb_a_rdy, 1);
        cb_pop_front(cb_a_rdy, 1);
        pass(row, a, IC, d_blocks, d_slot, IC);
    }
    noc_async_atomic_barrier();
}
