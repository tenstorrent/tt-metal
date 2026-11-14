// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// X40 TP32 routed experts, NCRISC of bank core b: streams the bank's weight blocks of the ACTIVE experts into the ring
// cb_w.
//   phase A: per active expert one block of 96 tiles (gate/up K slice, [24 k][g0 g1 u0 u1]) = 4 reads of 24 tiles
//   phase C: for sub-block s = 0..5, per active expert one block of 8 tiles (down [k 2][n 4]) = 1 read
// CT: 0 cb_w, 1 cb_rt (route staging page), 2 T, 3 ring tiles (multiple of 96 and 8), 4 tile bytes, 5 nsb, then
// TensorAccessorArgs(route). RT: 0 bank id, 1 vc. Common: 0 w01 addr, 1 w2 addr, 2 route addr.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tp_route.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rt = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t RING = get_compile_time_arg_val(3);
    constexpr uint32_t TB = get_compile_time_arg_val(4);
    constexpr uint32_t NSB = get_compile_time_arg_val(5);
    constexpr uint32_t DBG = get_compile_time_arg_val(6);
    constexpr auto r_args = TensorAccessorArgs<7>();
    constexpr uint32_t NTRID = 3;

    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w01_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w2_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_common_arg_val<uint32_t>(2);

    // route -> plan
    const auto r_acc = TensorAccessor(r_args, route_addr, T * 64);
    const uint32_t rt_l1 = get_write_ptr(cb_rt);
    noc_async_read(r_acc.get_noc_addr(0), rt_l1, T * 64);
    noc_async_read_barrier();
    uint16_t active[128];
    const uint32_t na = tp_build_plan(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_l1), T, active);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    auto& cbi = get_local_cb_interface(cb_w);
    const uint32_t ring_base = cbi.fifo_limit - cbi.fifo_size;
    const uint32_t ring_end = ring_base + RING * TB;
    uint32_t wr = ring_base;
    uint32_t pend_tiles = 0, pend_trid = 0, trid = 1;

    // one block: npages reads of page_bytes from `base` + off, contiguous in L1
    auto block = [&](uint64_t base, uint32_t off, uint32_t npages, uint32_t page_bytes) {
        const uint32_t tiles = npages * page_bytes / TB;
        cb_reserve_back(cb_w, pend_tiles + tiles);
        noc_async_read_set_trid(trid);
        for (uint32_t i = 0; i < npages; ++i) {
            noc_async_read_one_packet_with_state_with_trid(base, off + i * page_bytes, wr + i * page_bytes, trid);
        }
        wr += tiles * TB;
        if (wr >= ring_end) {
            wr = ring_base;
        }
        if (pend_tiles) {
            noc_async_read_barrier_with_trid(pend_trid);
            cb_push_back(cb_w, pend_tiles);
        }
        pend_tiles = tiles;
        pend_trid = trid;
        trid = (trid == NTRID) ? 1 : trid + 1;
    };
    auto flush = [&]() {
        if (pend_tiles) {
            noc_async_read_barrier_with_trid(pend_trid);
            cb_push_back(cb_w, pend_tiles);
            pend_tiles = 0;
        }
    };

    {                                     // phase A
        constexpr uint32_t PA = 24 * TB;  // 13824 B
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, w01_addr);
        noc_async_read_one_packet_set_state<true>(base, PA, vc);
        for (uint32_t a = 0; a < na; ++a) {
            block(base, (uint32_t)active[a] * 4 * PA, 4, PA);
        }
        flush();
    }
    if (DBG != 1 && DBG != 2) {          // phase C
        constexpr uint32_t PC = 8 * TB;  // 4608 B
        const uint64_t base = get_noc_addr_from_bank_id<true>(bank, w2_addr);
        noc_async_read_one_packet_set_state<true>(base, PC, vc);
        for (uint32_t s = 0; s < NSB; ++s) {
            for (uint32_t a = 0; a < na; ++a) {
                block(base, (s * 256 + (uint32_t)active[a]) * PC, 1, PC);
            }
        }
        flush();
    }
    noc_async_read_barrier();
}
