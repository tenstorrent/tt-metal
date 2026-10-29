// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill engine MLP stack, weight mover (NCRISC, NOC1); see mlp_compute.cpp for the layout. Grid column c
// computes output columns [c * N, (c + 1) * N) of each matmul, walking its weight rows K block by K block
// (a block is kb x N tiles, K-major). One column core reads a block from DRAM and multicasts it into every
// column core's weight CB (itself included); receivers reserve first and count themselves ready on it. The
// down weights land in u's memory, so they wait for cb_u_free.
//
// Compile-time args: 0 R, 1 Q, 2 IC, 3 W, 4 HC, 5 GU blocks per owner, 6 GU slot, 7 D blocks per owner,
// 8 D slot, 9 layers, 10 GU ring blocks, 11 D ring blocks (both >= 2). Runtime args: 0 grid row r, 1 grid
// column c, 2 column rectangle noc x0, 3 y0, 4 x1, 5 y1 (as NOC1 coordinates), then R x (x, y) of the
// column's cores, then per layer the G, U, D addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_gu = 1, cb_dw = 2, cb_u_free = 26;
constexpr uint32_t sem_ready = 2, sem_valid = 3;
// holds 1: the flag source, so the sender's own flag is set only by the multicast, after its block. The
// flag goes through the data's command buffer and VC, so it lands after the data.
constexpr uint32_t sem_one = 7;

constexpr uint32_t R = get_compile_time_arg_val(0);
constexpr uint32_t Q = get_compile_time_arg_val(1);
constexpr uint32_t IC = get_compile_time_arg_val(2);
constexpr uint32_t W = get_compile_time_arg_val(3);
constexpr uint32_t HC = get_compile_time_arg_val(4);
constexpr uint32_t gu_blocks = get_compile_time_arg_val(5);
constexpr uint32_t gu_slot = get_compile_time_arg_val(6);
constexpr uint32_t d_blocks = get_compile_time_arg_val(7);
constexpr uint32_t d_slot = get_compile_time_arg_val(8);
constexpr uint32_t layers = get_compile_time_arg_val(9);
constexpr uint32_t gu_ring = get_compile_time_arg_val(10);
constexpr uint32_t d_ring = get_compile_time_arg_val(11);

constexpr uint32_t kPacket = 4096;
constexpr uint32_t banks = R;  // one DRAM bank per grid row

struct Col {
    uint32_t r, c, x0, y0, x1, y1, vc;
    uint64_t mcast(uint32_t addr) const { return get_noc_multicast_addr(x0, y0, x1, y1, addr); }
    void inc(uint32_t q, uint32_t addr) const {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(6 + 2 * q), get_arg_val<uint32_t>(7 + 2 * q), addr), 1);
    }
};

// N output tiles per row; K = Q * per rows, owners' blocks of `slot` rows, the last one narrower. The ring
// holds `ring` blocks from `base`. Block b is sent by column row b % R; the sender of block b + 1 reads it
// from DRAM while block b is multicast. Block b of column c is contiguous in DRAM bank (b * Q + c) % banks
// at ((b * Q + c) / banks) * slot * N tiles, read in packets alternating NOC0 / NOC1.
template <uint32_t cb, uint32_t N>
void stream(
    const Col& col,
    uint32_t addr,
    uint32_t blocks_per_owner,
    uint32_t slot,
    uint32_t per,
    uint32_t base,
    uint32_t ring) {
    constexpr uint32_t tile = get_tile_size(cb);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));
    const uint32_t ready_addr = get_semaphore(sem_ready), valid_addr = get_semaphore(sem_valid);
    const uint32_t blk = slot * N, bytes = blk * tile, n = Q * blocks_per_owner;
    auto tiles = [&](uint32_t b) {
        const uint32_t i = b % blocks_per_owner;
        return (i + 1 < blocks_per_owner ? slot : per - slot * (blocks_per_owner - 1)) * N;
    };
    auto issue = [&](uint32_t b, uint32_t dst) {
        const uint32_t trid = (b & 1) + 1, g = b * Q + col.c;
        noc_async_read_set_trid(trid, 0);
        noc_async_read_set_trid(trid, 1);
        uint32_t src = (g / banks) * bytes, l1 = dst, left = tiles(b) * tile;
        for (uint32_t k = 0; left > 0; k++) {
            const uint32_t size = left < kPacket ? left : kPacket;
            const uint8_t noc = k & 1;
            noc_async_read_one_packet_set_state<true>(
                get_noc_addr_from_bank_id<true>(g % banks, addr, noc), size, col.vc, noc);
            noc_async_read_one_packet_with_state_with_trid(addr, src, l1, trid, noc);
            src += size;
            l1 += size;
            left -= size;
        }
    };
    auto next_of = [&](uint32_t dst) { return dst + bytes == base + ring * bytes ? base : dst + bytes; };

    uint32_t dst = get_write_ptr(cb);
    if (col.r == 0) {
        cb_reserve_back(cb, blk);
        issue(0, dst);
    }
    for (uint32_t b = 0; b < n; b++) {
        const uint32_t sender = b % R;
        cb_reserve_back(cb, blk);
        col.inc(sender, ready_addr);
        if (col.r == sender) {
            noc_async_read_barrier_with_trid((b & 1) + 1, 0);
            noc_async_read_barrier_with_trid((b & 1) + 1, 1);
            noc_semaphore_wait(ready, R);
            noc_semaphore_set(ready, 0);
            noc_async_write_multicast_loopback_src(dst, col.mcast(dst), tiles(b) * tile, R);
            noc_async_writes_flushed();
        }
        if (b + 1 < n && col.r == (b + 1) % R) {
            cb_reserve_back(cb, 2 * blk);
            issue(b + 1, next_of(dst));
        }
        if (col.r == sender) {
            noc_async_write_multicast_loopback_src(get_semaphore(sem_one), col.mcast(valid_addr), 4, R);
            noc_async_writes_flushed();
        }
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
        cb_push_back(cb, blk);
        dst = next_of(dst);
    }
    noc_async_write_barrier();
}
}  // namespace

void kernel_main() {
    const Col col{
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(1) & 3};
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 1);
    constexpr uint32_t addrs = 6 + 2 * R;
    const uint32_t gu_base = get_write_ptr(cb_gu), d_base = get_write_ptr(cb_dw);
    for (uint32_t l = 0; l < layers; l++) {
        stream<cb_gu, IC>(col, get_arg_val<uint32_t>(addrs + 3 * l), gu_blocks, gu_slot, HC, gu_base, gu_ring);
        stream<cb_gu, IC>(col, get_arg_val<uint32_t>(addrs + 1 + 3 * l), gu_blocks, gu_slot, HC, gu_base, gu_ring);
        cb_wait_front(cb_u_free, 1);
        cb_pop_front(cb_u_free, 1);
        stream<cb_dw, W>(col, get_arg_val<uint32_t>(addrs + 2 + 3 * l), d_blocks, d_slot, IC, d_base, d_ring);
    }
    noc_async_atomic_barrier();
}
