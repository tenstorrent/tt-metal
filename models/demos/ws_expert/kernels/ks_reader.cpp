// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert K-split groups NCRISC (NOC1). Core (C, Y) of a group (j = 10 C + Y). A non-blocking loop over:
//   QW weights: as the 1 x 20 groups (injector slices land in the 2-slot rings, WARR; credits back).
//   QA (column 1): this core's tilized half of x row s (CB 8) -> half C of slot s % NXR of its own and its partner's x
//      ring (NOC1 west: column 1 -> 0), once both consumed row s - NXR.
//   QB x ring: push row s (CB 0, KSL tiles) once both halves' flags read >= s + 1; when compute pops rows, write the
//      count into XCRED word 0 here and word 1 at the partner.
//   QL landing: push row q (CB 4, 20 tiles) once all 10 contributors' flags of slot q % D read >= q + 1; when compute
//      pops rows, write the count into LCRED word Y here (the BRISC credits the rows below) and at the rows above.
//   QU partials, rows above (NOC1): once the BRISC published PAVAIL >= q + 1, row q's contribution to each owner
//      o < Y (landing slot q % D, contributor Y; flag q + 1), once o consumed row q - D; then PDONE = q + 1.
//   QH h ring / QHL column 1 h leader multicasts / QY down outputs: as the 1 x 20 groups.
// RT: 0 j, 1 g, 2 nexp_g, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 injector x, 7 injector y,
//     8 credit offset at the injector, 9 stamps, 10 out addr
// CT: 0 KSL, 1 KTD, 2 DCOLS, 3 GC, 4 G, 5 NS, 6 hflags, 7-10 group rect for NOC1, 11 counts L1, 12 icred, 13 hring,
//     14 h tile, 15 warr, 16 RMAX, 17 HT, 18 NXR, 19 xring, 20 xflags, 21 xcred, 22 x tile, 23 D, 24 land,
//     25 lflags, 26 lcred, 27 p tile, 28 scratch, 29 PD, 30 partial ring (CB 18), then TensorAccessorArgs(out)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t land_cb = tt::CBIndex::c_4;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t p_cb = tt::CBIndex::c_18;
constexpr uint32_t SEM_HARR = 1, SEM_SCRH = 5;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0), KTD = get_compile_time_arg_val(1);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(2), GC = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4), NS = get_compile_time_arg_val(5);
    constexpr uint32_t hflags = get_compile_time_arg_val(6);
    constexpr uint32_t mx0 = get_compile_time_arg_val(7), my0 = get_compile_time_arg_val(8);
    constexpr uint32_t mx1 = get_compile_time_arg_val(9), my1 = get_compile_time_arg_val(10);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(11), icred = get_compile_time_arg_val(12);
    constexpr uint32_t hring = get_compile_time_arg_val(13), htile = get_compile_time_arg_val(14);
    constexpr uint32_t warr_addr = get_compile_time_arg_val(15);
    constexpr uint32_t RMAX = get_compile_time_arg_val(16), HT = get_compile_time_arg_val(17);
    constexpr uint32_t NXR = get_compile_time_arg_val(18), xring = get_compile_time_arg_val(19);
    constexpr uint32_t xflags = get_compile_time_arg_val(20), xcred = get_compile_time_arg_val(21);
    constexpr uint32_t xtile = get_compile_time_arg_val(22), D = get_compile_time_arg_val(23);
    constexpr uint32_t land = get_compile_time_arg_val(24), lflags = get_compile_time_arg_val(25);
    constexpr uint32_t lcred = get_compile_time_arg_val(26), ptile = get_compile_time_arg_val(27);
    constexpr uint32_t scr = get_compile_time_arg_val(28);
    constexpr uint32_t PD = get_compile_time_arg_val(29), pring = get_compile_time_arg_val(30);
    constexpr auto oargs = TensorAccessorArgs<31>();
    constexpr uint32_t HALF = GC / 2, KH = KSL / 2, NW = 2 * HALF, LT = 2 * HALF;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS;
#ifdef KS_XALT
    constexpr uint32_t XP = KSL, XSTEP = 2;
#else
    constexpr uint32_t XP = KH, XSTEP = 1;
#endif

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint32_t ix = get_arg_val<uint32_t>(6), iy = get_arg_val<uint32_t>(7), ioff = get_arg_val<uint32_t>(8);
    const uint32_t C = J / HALF, Y = J % HALF;
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    auto rows_of = [&](uint32_t n) { return (counts[16 + g + n * G] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

    // ---- QW (as the 1 x 20 groups)
    uint32_t w_n = 0, w_freed_sent = 0, gu_pages = 0, dn_pages = 0;
    volatile tt_l1_ptr uint32_t* warr = W(warr_addr);
    const uintptr_t gu_ack = (uintptr_t)get_cb_tiles_acked_ptr(w_cb), dn_ack = (uintptr_t)get_cb_tiles_acked_ptr(wd_cb);
    uint16_t gu_last = (uint16_t)reg_read(gu_ack), dn_last = (uint16_t)reg_read(dn_ack);
    const uint64_t cred_dst = get_noc_addr(ix, iy, icred + ioff);
    auto qw = [&]() -> bool {
        bool did = false;
        invalidate_l1_cache();
        if (w_n < nexp_g && warr[(w_n & 1) * 4] >= w_n / 2 + 1 && cb_pages_reservable_at_back(w_cb, WGU) &&
            cb_pages_reservable_at_back(wd_cb, WDN)) {
            cb_reserve_back(w_cb, WGU);
            cb_push_back(w_cb, WGU);
            cb_reserve_back(wd_cb, WDN);
            cb_push_back(wd_cb, WDN);
            ++w_n;
            did = true;
        }
        const uint16_t a = (uint16_t)reg_read(gu_ack), b = (uint16_t)reg_read(dn_ack);
        gu_pages += (uint16_t)(a - gu_last);
        dn_pages += (uint16_t)(b - dn_last);
        gu_last = a;
        dn_last = b;
        uint32_t freed = gu_pages / WGU;
        if (dn_pages / WDN < freed) {
            freed = dn_pages / WDN;
        }
        if (freed > w_freed_sent) {
            noc_inline_dw_write(cred_dst, freed);
            w_freed_sent = freed;
            did = true;
        }
        return did;
    };

    // ---- QA: column 1's x halves (NOC1 west)
    const uint32_t px = cxs[1 - C], py = ny0 + Y, mx = cxs[C];
    uint32_t aq = XSTEP == 2 ? C : 0;
    auto qa = [&]() -> bool {
        if (C != 1 || aq >= total_rows || !cb_pages_available_at_front(xst_cb, XP)) {
            return false;
        }
        if (aq >= NXR) {
            invalidate_l1_cache();
            if (*W(xcred) < aq - NXR + 1 || *W(xcred + 16) < aq - NXR + 1) {
                return false;
            }
        }
        const uint32_t dst = xring + ((aq % NXR) * KSL + (XSTEP == 2 ? 0 : C * KH)) * xtile;
        const uint32_t fl = xflags + ((aq % NXR) * 2 + C) * 16;
        *W(scr + 2 * 16) = aq + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        noc_async_write(src, get_noc_addr(mx, py, dst), XP * xtile);
        noc_async_write(scr + 2 * 16, get_noc_addr(mx, py, fl), 4);
        noc_async_write(src, get_noc_addr(px, py, dst), XP * xtile);
        noc_async_write(scr + 2 * 16, get_noc_addr(px, py, fl), 4);
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, XP);
        aq += XSTEP;
        return true;
    };

    // ---- QB: x ring pushes + credits
    uint32_t b_s = 0, x_pages = 0, x_cred = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_rows && cb_pages_reservable_at_back(xr_cb, KSL)) {
            invalidate_l1_cache();
            const uint32_t f = xflags + (b_s % NXR) * 2 * 16;
            const bool lo = XSTEP == 2 ? (b_s % 2 == 1 || *W(f) >= b_s + 1) : *W(f) >= b_s + 1;
            const bool hi = XSTEP == 2 ? (b_s % 2 == 0 || *W(f + 16) >= b_s + 1) : *W(f + 16) >= b_s + 1;
            if (lo && hi) {
                cb_reserve_back(xr_cb, KSL);
                cb_push_back(xr_cb, KSL);
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_pages += (uint16_t)(a - x_last);
        x_last = a;
        if (x_pages / KSL > x_cred) {
            x_cred = x_pages / KSL;
            *W(xcred) = x_cred;
            noc_inline_dw_write(get_noc_addr(px, py, xcred + 16), x_cred);
            did = true;
        }
        return did;
    };

    // ---- QL: landing pushes + credits (own word; the rows above)
    uint32_t l_q = 0, l_pages = 0, l_cons = 0;
    const uintptr_t lacked = (uintptr_t)get_cb_tiles_acked_ptr(land_cb);
    uint16_t l_last = (uint16_t)reg_read(lacked);
    auto ql = [&]() -> bool {
        bool did = false;
        if (l_q < total_rows && cb_pages_reservable_at_back(land_cb, LT)) {
            invalidate_l1_cache();
            bool all = true;
            for (uint32_t k = 0; k < HALF && all; ++k) {
                all = *W(lflags + ((l_q % D) * HALF + k) * 16) >= l_q + 1;
            }
            if (all) {
                cb_reserve_back(land_cb, LT);
                cb_push_back(land_cb, LT);
                ++l_q;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(lacked);
        l_pages += (uint16_t)(a - l_last);
        l_last = a;
        if (l_pages / LT > l_cons) {
            l_cons = l_pages / LT;
            *W(lcred + Y * 16) = l_cons;
            for (uint32_t o = 0; o < Y; ++o) {
                noc_inline_dw_write(get_noc_addr(mx, ny0 + o, lcred + Y * 16), l_cons);
            }
            did = true;
        }
        return did;
    };

    // ---- QU: partials to the rows above
    uint32_t uq = 0, uo = 0;
    auto qu = [&]() -> bool {
        if (uq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(scr + 4 * 16) < uq + 1) {  // PAVAIL
            return false;
        }
        if (uo == 0) {
            *W(scr + 3 * 16) = uq + 1;  // this RISC's partial flag source
        }
        const uint32_t src = pring + (uq % PD) * NW * ptile, s = uq % D;  // (CB 18: PD whole rows)
        bool did = false;
        while (uo < Y) {
            if (uq >= D && *W(lcred + uo * 16) < uq - D + 1) {
                return did;
            }
            const uint32_t oy = ny0 + uo;
            noc_async_write(src + 2 * uo * ptile, get_noc_addr(mx, oy, land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 3 * 16, get_noc_addr(mx, oy, lflags + (s * HALF + Y) * 16), 4);
            ++uo;
            did = true;
        }
        noc_async_writes_flushed();
        *W(scr + 5 * 16) = uq + 1;  // PDONE
        ++uq;
        uo = 0;
        return true;
    };

    // ---- QH: h ring rows
    uint32_t h_q = 0;
    auto qh = [&]() -> bool {
        if (h_q >= total_rows || !cb_pages_reservable_at_back(hr_cb, KTD)) {
            return false;
        }
        invalidate_l1_cache();
        for (uint32_t c = 0; c < 2; ++c) {
            if (*W(hflags + ((h_q % NS) * 2 + c) * 16) < h_q + 1) {
                return false;
            }
        }
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
        ++h_q;
        return true;
    };
    // ---- QHL: column 1's h leader multicasts
    uint32_t led = 0, lq = Y;
    const uint32_t harr = get_semaphore(SEM_HARR), hscr_addr = get_semaphore(SEM_SCRH);
    auto qhl = [&]() -> bool {
        if (C != 1 || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(harr) < HALF * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + C * HALF) * htile;
        *W(hscr_addr) = lq + 1;
        noc_async_write_multicast_loopback_src(
            part, get_noc_multicast_addr(mx0, my0, mx1, my1, part), HALF * htile, GC, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            hscr_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, hflags + ((lq % NS) * 2 + C) * 16), GC);
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };
    // ---- QY: down outputs of the group's last expert to DRAM
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(10), obytes);
    uint32_t drained = 0, d_n = 0, d_r = 0;
    auto qy = [&]() -> bool {
        if (drained >= total_rows || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        while (d_r >= rows_of(d_n)) {
            ++d_n;
            d_r = 0;
        }
        if (d_n + 1 == nexp_g) {
            uint32_t l1 = get_read_ptr(y_cb);
            const uint32_t row = g * RMAX + d_r;
            for (uint32_t t = 0; t < DCOLS; ++t, l1 += obytes) {
                noc_async_write_page(row * HT + J * DCOLS + t, os, l1);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(y_cb, DCOLS);
        ++d_r;
        ++drained;
        return true;
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t tq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    while (true) {
        bool did[8];
        uint32_t t[9];
        t[0] = wall();
        did[0] = qw();
        t[1] = wall();
        did[1] = qa();
        t[2] = wall();
        did[2] = qb();
        t[3] = wall();
        did[3] = ql();
        t[4] = wall();
        did[4] = qu();
        t[5] = wall();
        did[5] = qh();
        t[6] = wall();
        did[6] = qhl();
        t[7] = wall();
        did[7] = qy();
        t[8] = wall();
        bool any = false;
        for (uint32_t i = 0; i < 8; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        if (!any && w_n >= nexp_g && w_freed_sent >= nexp_g && (C != 1 || aq >= total_rows) && b_s >= total_rows &&
            x_cred >= total_rows && l_q >= total_rows && l_cons >= total_rows && uq >= total_rows &&
            h_q >= total_rows && (C != 1 || lq >= total_rows) && drained >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* st = W(get_arg_val<uint32_t>(9));
    for (uint32_t i = 0; i < 8; ++i) {
        st[470 + i] = tq[i];
    }
}
