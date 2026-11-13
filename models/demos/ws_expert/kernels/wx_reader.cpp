// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert x-streaming NCRISC (NOC1). Bench preload: this core's gate/up pairs (CB 1) and down columns (CB 3),
// resident. Then a non-blocking loop over in-order queues:
//   QA x multicast: this core's tilized x blocks (CB 8) -> slot (s % NX) of every cluster core's x ring (CB 0) as a
//      linked multicast closed by the slot's flag word (s + 1); a slot is reused once every cluster core popped its
//      previous block: each consumer writes its popped-block count into its own word at the producer of block c + NX
//      when it pops block c (monotone, so a plain write; a shared counter could add fast consumers' later credits
//      to a slot a slow consumer still reads), and block s waits until all GC words reach s - NX + 1.
//   QB x ring: push block s to compute once its flag reads >= s + 1; credit each block compute pops (CB 0's acked
//      counter) to the producer of the block that reuses its slot.
//   QC program order: PRE_L / PRE_R partials (push once arrived) and DN ops' h ring rows (push once every pair
//      group's flag reads >= row + 1); partial ring slots are granted eagerly.
// RT: 0 w addr, 1 wd addr, 2 has_left, 3 right x, 4 right y, 5 J, 6 nop, 7 has_right, 8 left x, 9 left y,
//     10 NB (own x blocks an expert), 11 NBE (x blocks an expert, all producers), then GC x (noc x, noc y) of the
//     cluster's cores, then per op: flags, then own block stream indices (in the expert) [NB]
// CT: 0 KT, 1 GW, 2 WCOLS, 3 NEXP, 4 KTD, 5 DCOLS, 6 HT, 7 NS, 8 GC, 9 hflags, 10 BK, 11 NX, 12 xring, 13 xflags,
//     14-17 column rect for NOC1 (start = the NOC0 end), 18 x tile bytes, 19 xprog (GC progress words, 16 B apart),
//     then TensorAccessorArgs(w), (wd)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t pinl_cb = tt::CBIndex::c_5;
constexpr uint32_t pinr_cb = tt::CBIndex::c_6;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3, SEM_XCRED = 5, SEM_SCR2 = 6;
constexpr uint32_t F_PRE_L = 1, F_PRE_R = 2, F_SEND = 12, F_DN = 16, F_TIL = 32 | 64;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr uint32_t WCOLS = get_compile_time_arg_val(2);
    constexpr uint32_t NEXP = get_compile_time_arg_val(3);
    constexpr uint32_t KTD = get_compile_time_arg_val(4);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(5);
    constexpr uint32_t HT = get_compile_time_arg_val(6);
    constexpr uint32_t NS = get_compile_time_arg_val(7);
    constexpr uint32_t GC = get_compile_time_arg_val(8);
    constexpr uint32_t hflags = get_compile_time_arg_val(9);
    constexpr uint32_t BK = get_compile_time_arg_val(10);
    constexpr uint32_t NX = get_compile_time_arg_val(11);
    constexpr uint32_t xring = get_compile_time_arg_val(12);
    constexpr uint32_t xflags = get_compile_time_arg_val(13);
    constexpr uint32_t mx0 = get_compile_time_arg_val(14), my0 = get_compile_time_arg_val(15);
    constexpr uint32_t mx1 = get_compile_time_arg_val(16), my1 = get_compile_time_arg_val(17);
    constexpr uint32_t xtile = get_compile_time_arg_val(18);
    constexpr uint32_t xprog = get_compile_time_arg_val(19);
    constexpr auto wargs = TensorAccessorArgs<20>();
    constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t ncol = GC;

    const uint32_t waddr = get_arg_val<uint32_t>(0), daddr = get_arg_val<uint32_t>(1);
    const uint32_t has_left = get_arg_val<uint32_t>(2);
    const uint32_t rx = get_arg_val<uint32_t>(3), ry = get_arg_val<uint32_t>(4);
    const uint32_t J = get_arg_val<uint32_t>(5);
    const uint32_t nop = get_arg_val<uint32_t>(6);
    const uint32_t has_right = get_arg_val<uint32_t>(7);
    const uint32_t lx = get_arg_val<uint32_t>(8), ly = get_arg_val<uint32_t>(9);
    const uint32_t NB = get_arg_val<uint32_t>(10);
    const uint32_t NBE = get_arg_val<uint32_t>(11);
    constexpr uint32_t col_base = 12;
    const uint32_t op_base = col_base + 2 * GC;
    const uint32_t own_base = op_base + nop;

    {
        const uint32_t wb = get_tile_size(w_cb), db = get_tile_size(wd_cb);
        const auto ws = TensorAccessor(wargs, waddr, wb);
        const auto ds = TensorAccessor(dargs, daddr, db);
        cb_reserve_back(w_cb, KT * GW);
        uint32_t l1 = get_write_ptr(w_cb);
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
        cb_push_back(w_cb, KT * GW);
        cb_push_back(wd_cb, KTD * DCOLS);
    }

    // ---- partial rings (grants to the neighbour that sends towards us)
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

    // ---- QA: own x block multicasts
    const uint32_t total_own = NB * NEXP;
    uint32_t a_i = 0;
    const uint32_t scr_addr = get_semaphore(SEM_SCR2);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    auto qa = [&]() -> bool {
        if (a_i >= total_own || !cb_pages_available_at_front(xst_cb, BK)) {
            return false;
        }
        const uint32_t s = (a_i / NB) * NBE + get_arg_val<uint32_t>(own_base + a_i % NB);
        if (s >= NX) {
            invalidate_l1_cache();
            for (uint32_t c = 0; c < GC; ++c) {
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + c * 16) < s - NX + 1) {
                    return false;
                }
            }
        }
        const uint32_t slot = s % NX;
        *scr = s + 1;
        noc_async_write_multicast_loopback_src(
            get_read_ptr(xst_cb),
            get_noc_multicast_addr(mx0, my0, mx1, my1, xring + slot * BK * xtile),
            BK * xtile,
            ncol,
            true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + slot * 16), ncol);
        noc_async_write_barrier();
        cb_pop_front(xst_cb, BK);
        ++a_i;
        return true;
    };

    // ---- QB: x ring pushes + credits
    const uint32_t total_x = NBE * NEXP;
    uint32_t b_s = 0, x_credited = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    uint32_t x_popped_pages = 0;
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_x && cb_pages_reservable_at_back(xr_cb, BK)) {
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + (b_s % NX) * 16);
            invalidate_l1_cache();
            if (*f >= b_s + 1) {
                cb_reserve_back(xr_cb, BK);
                cb_push_back(xr_cb, BK);
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_popped_pages += (uint16_t)(a - x_last);
        x_last = a;
        while (x_credited < x_popped_pages / BK) {
            const uint32_t c = x_credited++;
            const uint32_t nxt = c + NX;  // the block that reuses c's slot
            if (nxt < total_x) {
                const uint32_t owner = (nxt % NBE) % GC;
                noc_inline_dw_write(
                    get_noc_addr(
                        get_arg_val<uint32_t>(col_base + 2 * owner),
                        get_arg_val<uint32_t>(col_base + 2 * owner + 1),
                        xprog + J * 16),
                    c + 1);
            }
            did = true;
        }
        return did;
    };

    // ---- QC: program order (partials in, h ring rows)
    uint32_t c_e = 0, c_p = 0, produced = 0, consumed = 0, e_start = 0;
    bool c_flush = false;
    auto push_row_ready = [&](uint32_t r) -> bool {
        const uint32_t slot = r % NS;
        invalidate_l1_cache();
        for (uint32_t j = 0; j < GC; ++j) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + (slot * GC + j) * 16) < r + 1) {
                return false;
            }
        }
        return true;
    };
    auto qc = [&]() -> bool {
        regrant1(rl);
        regrant1(rr);
        if (c_flush) {
            if (consumed < produced && push_row_ready(consumed) && cb_pages_reservable_at_back(hr_cb, KTD)) {
                cb_reserve_back(hr_cb, KTD);
                cb_push_back(hr_cb, KTD);
                ++consumed;
                return true;
            }
            return false;
        }
        const uint32_t fl = get_arg_val<uint32_t>(op_base + c_p);
        bool done = false;
        if (fl & F_TIL) {
            done = true;
        } else if (fl & F_DN) {
            if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                if (push_row_ready(consumed) && cb_pages_reservable_at_back(hr_cb, KTD)) {
                    cb_reserve_back(hr_cb, KTD);
                    cb_push_back(hr_cb, KTD);
                    ++consumed;
                    done = true;
                }
            } else {
                done = true;
            }
        } else {
            Ring* g = (fl & F_PRE_L) ? &rl : ((fl & F_PRE_R) ? &rr : nullptr);
            if (g) {
                invalidate_l1_cache();
                if (*g->arr >= g->nrecv + 1 && cb_pages_reservable_at_back(g->cb, GW)) {
                    cb_reserve_back(g->cb, GW);
                    cb_push_back(g->cb, GW);
                    ++g->nrecv;
                    done = true;
                }
            } else {
                done = true;
            }
            if (done && !(fl & F_SEND)) {
                ++produced;
            }
        }
        if (done && ++c_p == nop) {
            c_p = 0;
            if (++c_e == NEXP) {
                c_flush = true;
            } else {
                e_start = produced;
            }
        }
        return done;
    };

    while (true) {
        const bool a = qa();
        const bool b = qb();
        const bool c = qc();
        if (!a && !b && !c && a_i >= total_own && b_s >= total_x && x_credited >= total_x && c_flush &&
            consumed >= produced) {
            break;
        }
    }
    while ((rl.on && rl.popped / GW < rl.nrecv) || (rr.on && rr.popped / GW < rr.nrecv)) {
        regrant1(rl);
        regrant1(rr);
    }
    noc_async_atomic_barrier();
}
