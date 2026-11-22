// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0

// Bank-local indexed expert matmul, reader (see tt/dram_sharded.py:BankLocalExperts).
// One worker core owns NW weight tile columns of every expert. Those columns live in ONE DRAM bank,
// laid out [expert][k][n] contiguously, so each selected expert is a single Kt*NW-tile read from
// that bank. The activation (all slots) is read once by the sender worker and multicast to the
// program's core rectangle (receivers handshake on two semaphores, as the mcast matmuls do).
// role: 0 idle (core only holds the CB the multicast lands in), 1 receiver, 2 sender.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_idx = 2;
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t NW = get_compile_time_arg_val(1);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(2);
    constexpr uint32_t IN0_TILES = get_compile_time_arg_val(3);
    constexpr uint32_t NUM_EXPERTS = get_compile_time_arg_val(4);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(5);
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(6);
    constexpr uint32_t SEM_VALID = get_compile_time_arg_val(7);
    constexpr auto in0_args = TensorAccessorArgs<8>();
    constexpr auto idx_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t role = get_arg_val<uint32_t>(6);
    if (role == 0) {
        return;
    }
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t expert_stride = get_arg_val<uint32_t>(2);
    const uint32_t worker_off = get_arg_val<uint32_t>(3);
    const uint32_t in0_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);

    const uint32_t in0_tile = get_tile_size(cb_in0);
    const uint32_t w_bytes = Kt * NW * get_tile_size(cb_in1);
    const auto idx = TensorAccessor(idx_args, idx_addr, IDX_PAGE);

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    noc_async_read_page(0, idx, idx_l1);

    cb_reserve_back(cb_in0, IN0_TILES);
    const uint32_t in0_l1 = get_write_ptr(cb_in0);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_VALID));
    if (role == 2) {
        const uint32_t mx0 = get_arg_val<uint32_t>(7), my0 = get_arg_val<uint32_t>(8);
        const uint32_t mx1 = get_arg_val<uint32_t>(9), my1 = get_arg_val<uint32_t>(10);
        const uint32_t num_dests = get_arg_val<uint32_t>(11);
        const uint32_t num_receivers = get_arg_val<uint32_t>(12);
        const auto in0 = TensorAccessor(in0_args, in0_addr, in0_tile);
        uint32_t l1 = in0_l1;
        for (uint32_t t = 0; t < IN0_TILES; ++t) {
            noc_async_read_page(t, in0, l1);
            l1 += in0_tile;
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
        noc_semaphore_wait(ready, num_receivers);
        noc_semaphore_set(ready, 0);
        noc_async_write_multicast(
            in0_l1, get_noc_multicast_addr(mx0, my0, mx1, my1, in0_l1), IN0_TILES * in0_tile, num_dests, true);
        noc_semaphore_set(valid, 1);
        noc_semaphore_set_multicast(
            get_semaphore(SEM_VALID), get_noc_multicast_addr(mx0, my0, mx1, my1, get_semaphore(SEM_VALID)), num_dests);
    } else {
        const uint32_t sx = get_arg_val<uint32_t>(7), sy = get_arg_val<uint32_t>(8);
        noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_READY)), 1);
        noc_async_read_barrier();
        noc_semaphore_wait(valid, 1);
        noc_semaphore_set(valid, 0);
    }
    cb_push_back(cb_in0, IN0_TILES);
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);

    for (uint32_t s = 0; s < SLOTS; ++s) {
        uint32_t e = ids[s];
        if (e >= NUM_EXPERTS) {
            e = NUM_EXPERTS - 1;  // device-resident ids: never read outside the weight buffer
        }
        cb_reserve_back(cb_in1, Kt * NW);
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, w_addr + e * expert_stride + worker_off);
        noc_async_read(src, get_write_ptr(cb_in1), w_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in1, Kt * NW);
    }
    if (role == 2) {
        noc_async_write_barrier();
        noc_semaphore_set(valid, 0);  // only after the multicast of it has left
    }
}
