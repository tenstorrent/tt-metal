// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert dynamic-M NCRISC (NOC1). Bench preload: this core's gate/up pairs (CB 1) and down columns (CB 3). Then a
// non-blocking loop over in-order queues, each following the plans (ws_plan.hpp, filled by the BRISC):
//   QA x multicast: this core's tilized x blocks (CB 8; plan's own blocks, global stream index s = the cluster's
//      blocks of earlier experts + local index) -> slot s % NX of every cluster core's x ring, linked payload + flag
//      (s + 1); waits until every cluster core's popped-block count (its word in XPROG) reaches s - NX + 1.
//   QB x ring: push block s once its flag reads >= s + 1; when compute pops block c, write c + 1 into this core's
//      XPROG word at the producer of block c + NX (one unicast; that block's expert and local index are found by
//      walking the plans' block counts, so the credit waits until that expert is planned). A column multicast of the
//      popped count per block instead (~230 small multicasts an expert a column) starved x: 64K cycles of x waits.
//   QC program order: PRE_L / PRE_R partials and DN ops' h ring rows; partial ring slots granted eagerly.
// Posts its done word (expert + 1 finished by all three queues) for the plan ring.
// RT: 0 w addr, 1 wd addr, 2 has_left, 3 right x, 4 right y, 5 J, 6 has_right, 7 left x, 8 left y, 9 column noc x,
//     10 column noc y of core 0 (core j at y0 + j)
// CT: 0 KT, 1 GW, 2 WCOLS, 3 NEXP, 4 KTD, 5 DCOLS, 6 HT, 7 NS, 8 GC, 9 hflags, 10 BK, 11 NX, 12 xring, 13 xflags,
//     14-17 column rect for NOC1 (start = the NOC0 end), 18 x tile bytes, 19 xprog, 20 plan buffer,
//     then TensorAccessorArgs(w), (wd)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"
#include "ws_plan.hpp"

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t pinl_cb = tt::CBIndex::c_5;
constexpr uint32_t pinr_cb = tt::CBIndex::c_6;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3, SEM_SCR2 = 6, SEM_SCR3 = 7;
using namespace ws_plan;

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
    constexpr uint32_t planbuf = get_compile_time_arg_val(20);
    constexpr auto wargs = TensorAccessorArgs<21>();
    constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t ncol = GC;

    const uint32_t waddr = get_arg_val<uint32_t>(0), daddr = get_arg_val<uint32_t>(1);
    const uint32_t has_left = get_arg_val<uint32_t>(2);
    const uint32_t rx = get_arg_val<uint32_t>(3), ry = get_arg_val<uint32_t>(4);
    const uint32_t J = get_arg_val<uint32_t>(5);
    const uint32_t has_right = get_arg_val<uint32_t>(6);
    const uint32_t lx = get_arg_val<uint32_t>(7), ly = get_arg_val<uint32_t>(8);
    const uint32_t col_x = get_arg_val<uint32_t>(9), col_y0 = get_arg_val<uint32_t>(10);  // NoC x / first y of the column

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
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(planbuf);
    volatile tt_l1_ptr uint32_t* ctrl = plan + CTRL;
    auto ready = [&](uint32_t e) -> volatile tt_l1_ptr uint32_t* {  // the plan of expert e, or null if not yet
        volatile tt_l1_ptr uint32_t* p = plan + (e % NSLOT) * SLOT_WORDS;
        invalidate_l1_cache();
        return p[H_READY] == e + 1 ? p : nullptr;
    };

    // ---- partial rings
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
    uint32_t a_e = 0, a_i = 0, a_base = 0;  // expert, own block index, cluster blocks of earlier experts
    const uint32_t scr_addr = get_semaphore(SEM_SCR2);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    auto qa = [&]() -> bool {
        while (a_e < NEXP) {  // advance past finished experts
            volatile tt_l1_ptr uint32_t* p = ready(a_e);
            if (!p) {
                return false;
            }
            if (a_i < p[H_NMY]) {
                break;
            }
            a_base += p[H_NBE];
            ++a_e;
            a_i = 0;
        }
        if (a_e >= NEXP || !cb_pages_available_at_front(xst_cb, BK)) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* p = plan + (a_e % NSLOT) * SLOT_WORDS;
        const uint32_t s = a_base + p[OFF_MY + 4 * a_i];
        if (s >= NX) {
            invalidate_l1_cache();
            for (uint32_t c = 0; c < GC; ++c) {
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + c * 16) < s - NX + 1) {
                    return false;
                }
            }
        }
        const uint32_t xs = s % NX;
        *scr = s + 1;
        noc_async_write_multicast_loopback_src(
            get_read_ptr(xst_cb),
            get_noc_multicast_addr(mx0, my0, mx1, my1, xring + xs * BK * xtile),
            BK * xtile,
            ncol,
            true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + xs * 16), ncol);
        noc_async_write_barrier();
        cb_pop_front(xst_cb, BK);
        ++a_i;
        return true;
    };

    // ---- QB: x ring pushes + credits to the producer of the block that reuses the slot
    uint32_t b_e = 0, b_i = 0, b_s = 0;  // expert, block index in it, global stream index
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    uint32_t x_popped_pages = 0, x_credited = 0;
    uint32_t w_e = 0, w_base = 0;  // credit walk: expert containing block x_credited + NX, its first global index
    auto qb = [&]() -> bool {
        bool did = false;
        while (b_e < NEXP) {
            volatile tt_l1_ptr uint32_t* p = ready(b_e);
            if (!p) {
                break;
            }
            if (b_i < p[H_NBE]) {
                break;
            }
            ++b_e;
            b_i = 0;
        }
        if (b_e < NEXP && ready(b_e) && cb_pages_reservable_at_back(xr_cb, BK)) {
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + (b_s % NX) * 16);
            invalidate_l1_cache();
            if (*f >= b_s + 1) {
                cb_reserve_back(xr_cb, BK);
                cb_push_back(xr_cb, BK);
                ++b_s;
                ++b_i;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_popped_pages += (uint16_t)(a - x_last);
        x_last = a;
        while (x_credited < x_popped_pages / BK) {
            const uint32_t nxt = x_credited + NX;
            bool known = false, past_end = false;
            while (true) {  // find nxt's expert (plans are made in order; an expert's slot outlives its blocks)
                if (w_e >= NEXP) {
                    past_end = true;
                    break;
                }
                volatile tt_l1_ptr uint32_t* p = ready(w_e);
                if (!p) {
                    break;
                }
                const uint32_t nbe = p[H_NBE];
                if (nxt < w_base + nbe) {
                    known = true;
                    const uint32_t owner = (nxt - w_base) % GC;
                    noc_inline_dw_write(
                        get_noc_addr(col_x, col_y0 + owner, xprog + J * 16),
                        x_credited + 1);
                    break;
                }
                w_base += nbe;
                ++w_e;
            }
            if (!known && !past_end) {
                break;  // (that expert is not planned yet: credit later)
            }
            ++x_credited;
            did = true;
        }
        return did;
    };

    // ---- QC: program order (partials in, h ring rows)
    uint32_t c_e = 0, c_p = 0, produced = 0, consumed = 0, e_start = 0;
    auto row_ready = [&](uint32_t r) -> bool {
        const uint32_t hs = r % NS;
        invalidate_l1_cache();
        for (uint32_t j = 0; j < GC; ++j) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + (hs * GC + j) * 16) < r + 1) {
                return false;
            }
        }
        return true;
    };
    auto push_row = [&]() -> bool {
        if (row_ready(consumed) && cb_pages_reservable_at_back(hr_cb, KTD)) {
            cb_reserve_back(hr_cb, KTD);
            cb_push_back(hr_cb, KTD);
            ++consumed;
            return true;
        }
        return false;
    };
    auto qc = [&]() -> bool {
        regrant1(rl);
        regrant1(rr);
        if (c_e >= NEXP) {
            return consumed < produced ? push_row() : false;
        }
        volatile tt_l1_ptr uint32_t* p = ready(c_e);
        if (!p) {
            return false;
        }
        bool done = false;
        if (c_p < p[H_NOP]) {
            const uint32_t fl = p[OFF_OP + 3 * c_p + 2];
            if (fl & DN) {
                if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                    done = push_row();
                } else {
                    done = true;
                }
            } else {
                Ring* g = (fl & PRE_L) ? &rl : ((fl & PRE_R) ? &rr : nullptr);
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
                if (done && !(fl & (SEND_L | SEND_R))) {
                    ++produced;
                }
            }
            if (done) {
                ++c_p;
            }
        }
        if (c_p >= p[H_NOP]) {
            c_p = 0;
            ++c_e;
            e_start = produced;
            done = true;
        }
        return done;
    };

    while (true) {
        const bool a = qa();
        const bool b = qb();
        const bool c = qc();
        uint32_t d = a_e < b_e ? a_e : b_e;
        d = d < c_e ? d : c_e;
        d = d < w_e ? d : w_e;  // (the credit walk still reads plan w_e)
        ctrl[DONE_NC] = d;
        if (!a && !b && !c && a_e >= NEXP && b_e >= NEXP && c_e >= NEXP && consumed >= produced) {
            // (credits still owed are for blocks past the last expert: none)
            break;
        }
    }
    while ((rl.on && rl.popped / GW < rl.nrecv) || (rr.on && rr.popped / GW < rr.nrecv)) {
        regrant1(rl);
        regrant1(rr);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
