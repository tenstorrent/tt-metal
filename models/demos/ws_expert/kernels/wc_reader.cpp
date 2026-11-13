// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert column-cluster reader (NCRISC, NOC1). Bench preload: the cluster's x store (XT tiles, CB 0), this core's
// gate/up pairs [KT][GW] (CB 1) and down columns [KTD][DCOLS] (CB 3), resident. Then in the compute's order:
//   PRE_L / PRE_R piece: wait for the left / right neighbour's partial (ARRIVED_L / _R), push it (CB 5 / CB 6).
//              Ring slots are granted eagerly: 2 up front, one more each time compute pops one (polled from the CB's
//              acked counter).
//   h piece q > 0: wait for every pair group's flag of row q - 1 (>= q), push that h ring row (CB 2) to compute.
// RT: 0 x addr, 1 w addr, 2 wd addr, 3 store, 4 has_left (neighbour), 5 right x, 6 right y, 7 J, 8 npiece,
//     9 has_right, 10 left x, 11 left y, then per op: flags (1 PRE_L, 2 PRE_R, 4 SEND_L, 8 SEND_R, 16 DN)
// CT: 0 KT, 1 GW, 2 XT, 3 WCOLS, 4 NEXP, 5 KTD, 6 DCOLS, 7 HT, 8 NS, 9 GC, 10 flags addr, then
//     TensorAccessorArgs(x), (w), (wd)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t x_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t pinl_cb = tt::CBIndex::c_5;
constexpr uint32_t pinr_cb = tt::CBIndex::c_6;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr uint32_t XT = get_compile_time_arg_val(2);
    constexpr uint32_t WCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t NEXP = get_compile_time_arg_val(4);
    constexpr uint32_t KTD = get_compile_time_arg_val(5);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(6);
    constexpr uint32_t HT = get_compile_time_arg_val(7);
    constexpr uint32_t NS = get_compile_time_arg_val(8);
    constexpr uint32_t GC = get_compile_time_arg_val(9);
    constexpr uint32_t flags = get_compile_time_arg_val(10);
    constexpr auto xargs = TensorAccessorArgs<11>();
    constexpr auto wargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    const uint32_t xaddr = get_arg_val<uint32_t>(0), waddr = get_arg_val<uint32_t>(1), daddr = get_arg_val<uint32_t>(2);
    const uint32_t store = get_arg_val<uint32_t>(3);
    const uint32_t has_left = get_arg_val<uint32_t>(4);
    const uint32_t rx = get_arg_val<uint32_t>(5), ry = get_arg_val<uint32_t>(6);
    const uint32_t has_right = get_arg_val<uint32_t>(9);
    const uint32_t lx = get_arg_val<uint32_t>(10), ly = get_arg_val<uint32_t>(11);
    const uint32_t J = get_arg_val<uint32_t>(7);
    const uint32_t npiece = get_arg_val<uint32_t>(8);

    {
        const uint32_t xb = get_tile_size(x_cb), wb = get_tile_size(w_cb), db = get_tile_size(wd_cb);
        const auto xs = TensorAccessor(xargs, xaddr, xb);
        const auto ws = TensorAccessor(wargs, waddr, wb);
        const auto ds = TensorAccessor(dargs, daddr, db);
        cb_reserve_back(x_cb, XT);
        uint32_t l1 = get_write_ptr(x_cb);
        for (uint32_t t = 0; t < XT; ++t, l1 += xb) {
            noc_async_read_page(store * XT + t, xs, l1);
        }
        cb_reserve_back(w_cb, KT * GW);
        l1 = get_write_ptr(w_cb);
        for (uint32_t k = 0; k < KT; ++k) {
            for (uint32_t t = 0; t < GW; ++t, l1 += wb) {
                noc_async_read_page(k * WCOLS + J * GW + t, ws, l1);
            }
        }
        cb_reserve_back(wd_cb, KTD * DCOLS);
        l1 = get_write_ptr(wd_cb);
        for (uint32_t k = 0; k < KTD; ++k) {
            for (uint32_t t = 0; t < DCOLS; ++t, l1 += db) {
                noc_async_read_page(k * HT + J * DCOLS + t, ds, l1);
            }
        }
        noc_async_read_barrier();
        cb_push_back(x_cb, XT);
        cb_push_back(w_cb, KT * GW);
        cb_push_back(wd_cb, KTD * DCOLS);
    }

    // Ring for partials from one side: grants go to that neighbour's GRANT semaphore for sending towards us.
    struct Ring {
        uint32_t cb;
        bool on;
        uint64_t grant_dst;
        volatile tt_l1_ptr uint32_t* arr;
        uintptr_t acked_ptr;
        uint16_t last_acked;
        uint32_t popped, granted, nrecv;
    };
    Ring rl{pinl_cb, has_left != 0, get_noc_addr(lx, ly, get_semaphore(SEM_GRANT_R)),
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARR_L)),
            (uintptr_t)get_cb_tiles_acked_ptr(pinl_cb), 0, 0, 0, 0};
    Ring rr{pinr_cb, has_right != 0, get_noc_addr(rx, ry, get_semaphore(SEM_GRANT_L)),
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ARR_R)),
            (uintptr_t)get_cb_tiles_acked_ptr(pinr_cb), 0, 0, 0, 0};
    rl.last_acked = (uint16_t)reg_read(rl.acked_ptr);
    rr.last_acked = (uint16_t)reg_read(rr.acked_ptr);
    auto regrant1 = [&](Ring& g) {
        if (!g.on) {
            return;
        }
        const uint16_t a = (uint16_t)reg_read(g.acked_ptr);
        g.popped += (uint16_t)(a - g.last_acked);
        g.last_acked = a;
        const uint32_t target = g.popped / GW + 2;
        if (g.granted < target) {
            noc_semaphore_inc(g.grant_dst, target - g.granted);
            g.granted = target;
        }
    };
    auto regrant = [&]() {
        regrant1(rl);
        regrant1(rr);
    };
    regrant();
    auto recv = [&](Ring& g) {
        invalidate_l1_cache();
        while (*g.arr < g.nrecv + 1) {
            regrant();
            invalidate_l1_cache();
        }
        cb_reserve_back(g.cb, GW);
        cb_push_back(g.cb, GW);
        ++g.nrecv;
    };

    uint32_t q = 0, consumed = 0;
    auto push_row = [&](uint32_t r) {  // h ring row r of the cluster's sequence
        const uint32_t slot = r % NS;
#ifndef WC_NOSYNC
        for (uint32_t j = 0; j < GC; ++j) {
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags + (slot * GC + j) * 16);
            invalidate_l1_cache();
            while (*f < r + 1) {
                regrant();
                invalidate_l1_cache();
            }
        }
#endif
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
    };
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t e_start = q;  // (DN_LAG: DN ops take rows of earlier experts or >= DN_LAG productions old)
        for (uint32_t p = 0; p < npiece; ++p) {
            const uint32_t fl = get_arg_val<uint32_t>(12 + p);
            if (fl & 16) {
                if (consumed < q && (!DN_LAG || consumed < e_start || consumed + DN_LAG < q)) {
                    push_row(consumed++);
                }
                continue;
            }
            if (fl & 1) {
                recv(rl);
            }
            if (fl & 2) {
                recv(rr);
            }
            if (!(fl & 12)) {
                ++q;
            }
        }
    }
    while (consumed < q) {
        push_row(consumed++);
    }
    while ((rl.on && rl.popped / GW < rl.nrecv) || (rr.on && rr.popped / GW < rr.nrecv)) {
        regrant();
    }
    noc_async_atomic_barrier();
}
