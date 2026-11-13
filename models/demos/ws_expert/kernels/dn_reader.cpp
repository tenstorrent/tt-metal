// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert down reader (BRISC, NOC0). Bench preload: the core's down weights [KTD][CW] into CB 1 (resident). Then
// per row q of the h ring (q = e * R + place): reserve the ring slot (compute freed row q - NS), tell the leader
// (CREDIT += 1), wait until every pair group's flag word of the slot reads >= q + 1, push the row to compute.
// Leader (LEADER defined): publishes CONS = c to every gate/up core once all ND down cores credited row c - 1.
// RT: 0 wd addr, 1 column tile base (c * CW)
// CT: 0 KTD, 1 CW, 2 NOUT (tiles a Wd row), 3 R, 4 NS, 5 GC, 6 NEXP, 7 flags addr, 8 leader x, 9 leader y,
//     10 ND, 11 NRECT (gate/up NoC rectangles, <= 2), 12-21 two rects (x0, y0, x1, y1, n), then TensorAccessorArgs(wd)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t h_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_1;
constexpr uint32_t SEM_CONS = 0, SEM_SCR = 3, SEM_CREDIT = 4;

constexpr uint32_t KTD = get_compile_time_arg_val(0);
constexpr uint32_t CW = get_compile_time_arg_val(1);
constexpr uint32_t NOUT = get_compile_time_arg_val(2);
constexpr uint32_t R = get_compile_time_arg_val(3);
constexpr uint32_t NS = get_compile_time_arg_val(4);
constexpr uint32_t GC = get_compile_time_arg_val(5);
constexpr uint32_t NEXP = get_compile_time_arg_val(6);
constexpr uint32_t flags = get_compile_time_arg_val(7);
constexpr uint32_t lx = get_compile_time_arg_val(8), ly = get_compile_time_arg_val(9);
constexpr uint32_t ND = get_compile_time_arg_val(10);
constexpr uint32_t NRECT = get_compile_time_arg_val(11);
constexpr uint32_t ax0 = get_compile_time_arg_val(12), ay0 = get_compile_time_arg_val(13);
constexpr uint32_t ax1 = get_compile_time_arg_val(14), ay1 = get_compile_time_arg_val(15), an = get_compile_time_arg_val(16);
constexpr uint32_t bx0 = get_compile_time_arg_val(17), by0 = get_compile_time_arg_val(18);
constexpr uint32_t bx1 = get_compile_time_arg_val(19), by1 = get_compile_time_arg_val(20), bn = get_compile_time_arg_val(21);
constexpr uint32_t NSEQ = NEXP * R;

#ifdef LEADER
uint32_t pub = 0;
FORCE_INLINE void pump() {
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_CREDIT));
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    invalidate_l1_cache();
    while (pub < NSEQ && *credit >= ND * (pub + 1)) {
        ++pub;
        noc_async_writes_flushed();  // the previous publish has left scr
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr) = pub;
        noc_semaphore_set_multicast(scr_addr, get_noc_multicast_addr(ax0, ay0, ax1, ay1, get_semaphore(SEM_CONS)), an);
        if constexpr (NRECT > 1) {
            noc_semaphore_set_multicast(
                scr_addr, get_noc_multicast_addr(bx0, by0, bx1, by1, get_semaphore(SEM_CONS)), bn);
        }
        invalidate_l1_cache();
    }
}
#else
FORCE_INLINE void pump() {}
#endif

void kernel_main() {
    constexpr auto wargs = TensorAccessorArgs<22>();
    const uint32_t waddr = get_arg_val<uint32_t>(0);
    const uint32_t col0 = get_arg_val<uint32_t>(1);
    const uint32_t wbytes = get_tile_size(wd_cb);
    const auto ws = TensorAccessor(wargs, waddr, wbytes);
    cb_reserve_back(wd_cb, KTD * CW);
    uint32_t l1 = get_write_ptr(wd_cb);
    for (uint32_t k = 0; k < KTD; ++k) {
        for (uint32_t t = 0; t < CW; ++t) {
            noc_async_read_page(k * NOUT + col0 + t, ws, l1);
            l1 += wbytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(wd_cb, KTD * CW);

    const uint64_t leader_credit = get_noc_addr(lx, ly, get_semaphore(SEM_CREDIT));
    volatile tt_l1_ptr uint32_t* prof = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2)) + 32;
    uint32_t t_res = 0, t_flag = 0;
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    // Credits: ring slot of seq s is free once compute popped seq s - NS. Credit every freed seq right away (polled
    // from the CB's acked-pages counter), not when the reader gets to it: NS rows stay in flight.
    const uintptr_t acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(h_cb);
    uint16_t last_acked = (uint16_t)reg_read(acked_ptr);
    uint32_t popped_pages = 0, credited = 0;
    auto credit = [&]() {
        const uint16_t a = (uint16_t)reg_read(acked_ptr);
        popped_pages += (uint16_t)(a - last_acked);
        last_acked = a;
        uint32_t target = popped_pages / KTD + NS;
        if (target > NSEQ) {
            target = NSEQ;
        }
        if (credited < target) {
            noc_semaphore_inc(leader_credit, target - credited);
            credited = target;
        }
    };
    for (uint32_t q = 0; q < NSEQ; ++q) {
        const uint32_t slot = q % NS;
        uint32_t a = wall();
        credit();
        uint32_t b = wall();
        for (uint32_t j = 0; j < GC; ++j) {
            volatile tt_l1_ptr uint32_t* f =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags + (slot * GC + j) * 16);
            invalidate_l1_cache();
            while (*f < q + 1) {
                credit();
                pump();
                invalidate_l1_cache();
            }
        }
        uint32_t c = wall();
        cb_reserve_back(h_cb, KTD);  // (free already: the producers wrote it after every down core freed it)
        cb_push_back(h_cb, KTD);
        t_res += wall() - c;
        t_flag += c - b;
    }
    prof[0] = t_res;
    prof[1] = t_flag;
#ifdef LEADER
    while (pub < NSEQ) {
        pump();
    }
    noc_async_write_barrier();
#endif
    noc_async_atomic_barrier();
}
