// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert 5 x 5 K-split groups, NCRISC (NOC1). Core (C, Y) of a group, j = 5 C + Y. A non-blocking loop over:
//   QW weights: the injector's slices land in the 2-slot rings (WARR word n % 2); credits back to the injector.
//   QB x ring: push row s (CB 0, KSL tiles) once its slot's flag reads >= s + 1; when compute pops rows, write the
//      count into XCRED word C of the 5 cores of row Y (its own locally).
//   QL (owners): push landing row q (CB 4, 10 tiles) once the 5 contributors' flags of slot q % D read >= q + 1; when
//      compute pops rows, the count into LCRED word Y here and at the owners above (the BRISC does the rows below).
//   QU partials to the owners o < Y (rows above, NOC1): once the BRISC published PAVAIL >= q + 1, tiles 2 o, 2 o + 1
//      of row q (CB 18 ring, by address) -> o's landing slot q % D at contributor Y (+ flag), once o consumed row
//      q - D; then PDONE = q + 1.
//   QH h ring: push row q (CB 2, KTD tiles) once the 5 column flags of slot q % NS read >= q + 1.
//   QHL (owners, odd columns): leader multicasts of the column's 4 h tiles (NOC1).
//   QY down outputs of the group's last expert to DRAM (DCOLS columns from DS).
// CT: 0-20 as g5_writer.cpp, 21-24 group rect for NOC1, 25 icred, 26 warr, 27 RMAX, 28 HT, 29 PD, 30 partial ring,
//     then TensorAccessorArgs(out)
// RT: 0 j, 1 g, 2 nexp_g, 3 injector x, 4 injector y, 5 credit offset at the injector, 6 stamps, 7 out addr, 8 DS,
//     9-13 column NOC x, 14-18 row NOC y
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t land_cb = tt::CBIndex::c_4;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t SEM_HARR = 1;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0), DCOLS = get_compile_time_arg_val(1);
    constexpr bool OWN = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t NXR = get_compile_time_arg_val(3);
    constexpr uint32_t xflags = get_compile_time_arg_val(5), xcred = get_compile_time_arg_val(6);
    constexpr uint32_t D = get_compile_time_arg_val(8);
    constexpr uint32_t land = get_compile_time_arg_val(9), lflags = get_compile_time_arg_val(10);
    constexpr uint32_t lcred = get_compile_time_arg_val(11), ptile = get_compile_time_arg_val(12);
    constexpr uint32_t scr = get_compile_time_arg_val(13), KTD = get_compile_time_arg_val(14);
    constexpr uint32_t G = get_compile_time_arg_val(15), NS = get_compile_time_arg_val(16);
    constexpr uint32_t hring = get_compile_time_arg_val(17), hflags = get_compile_time_arg_val(18);
    constexpr uint32_t htile = get_compile_time_arg_val(19), counts_l1 = get_compile_time_arg_val(20);
    constexpr uint32_t mx0 = get_compile_time_arg_val(21), my0 = get_compile_time_arg_val(22);
    constexpr uint32_t mx1 = get_compile_time_arg_val(23), my1 = get_compile_time_arg_val(24);
    constexpr uint32_t icred = get_compile_time_arg_val(25), warr_addr = get_compile_time_arg_val(26);
    constexpr uint32_t RMAX = get_compile_time_arg_val(27), HT = get_compile_time_arg_val(28);
    constexpr uint32_t PD = get_compile_time_arg_val(29), pring = get_compile_time_arg_val(30);
    constexpr auto oargs = TensorAccessorArgs<31>();
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS, GC = NR * NCOL;

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t ix = get_arg_val<uint32_t>(3), iy = get_arg_val<uint32_t>(4), ioff = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    const uint32_t DS = get_arg_val<uint32_t>(8);
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(9 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(14 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
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

    // ---- QW
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

    // ---- QB: x ring pushes + credits to the row
    uint32_t b_s = 0, x_pages = 0, x_cred = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_rows && cb_pages_reservable_at_back(xr_cb, KSL)) {
            invalidate_l1_cache();
            if (*W(xflags + (b_s % NXR) * 16) >= b_s + 1) {
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
            *W(xcred + C * 16) = x_cred;
            for (uint32_t t = 1; t < NCOL; ++t) {
                noc_inline_dw_write(get_noc_addr(cx[(C + t) % NCOL], ry[Y], xcred + C * 16), x_cred);
            }
            did = true;
        }
        return did;
    };

    // ---- QL (owners): landing pushes + credits (own word; the owners above)
    uint32_t l_q = 0, l_pages = 0, l_cons = 0;
    const uintptr_t lacked = (uintptr_t)get_cb_tiles_acked_ptr(land_cb);
    uint16_t l_last = (uint16_t)reg_read(lacked);
    auto ql = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        bool did = false;
        if (l_q < total_rows && cb_pages_reservable_at_back(land_cb, LT)) {
            invalidate_l1_cache();
            bool all = true;
            for (uint32_t k = 0; k < NR && all; ++k) {
                all = *W(lflags + ((l_q % D) * NR + k) * 16) >= l_q + 1;
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
                noc_inline_dw_write(get_noc_addr(cx[C], ry[o], lcred + Y * 16), l_cons);
            }
            did = true;
        }
        return did;
    };

    // ---- QU: partials to the owners above
    const uint32_t nup = Y < NOWN ? Y : NOWN;
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
        const uint32_t src = pring + (uq % PD) * NW * ptile, s = uq % D;
        bool did = false;
        while (uo < nup) {
            if (uq >= D && *W(lcred + uo * 16) < uq - D + 1) {
                return did;
            }
            noc_async_write(
                src + 2 * uo * ptile, get_noc_addr(cx[C], ry[uo], land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 3 * 16, get_noc_addr(cx[C], ry[uo], lflags + (s * NR + Y) * 16), 4);
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
        for (uint32_t c = 0; c < NCOL; ++c) {
            if (*W(hflags + ((h_q % NS) * NCOL + c) * 16) < h_q + 1) {
                return false;
            }
        }
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
        ++h_q;
        return true;
    };
    // ---- QHL (owners, odd columns): leader multicasts on NOC1
    const bool leads = OWN && C % 2 == 1;
    uint32_t led = 0, lq = Y;
    const uint32_t harr = get_semaphore(SEM_HARR);
    auto qhl = [&]() -> bool {
        if (!leads || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(harr) < NOWN * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + PC * C) * htile;
        *W(scr + 7 * 16) = lq + 1;
        noc_async_write_multicast_loopback_src(
            part, get_noc_multicast_addr(mx0, my0, mx1, my1, part), PC * htile, GC, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr + 7 * 16, get_noc_multicast_addr(mx0, my0, mx1, my1, hflags + ((lq % NS) * NCOL + C) * 16), GC);
        noc_async_writes_flushed();
        ++led;
        lq += NOWN;
        return true;
    };
    // ---- QY: down outputs of the group's last expert
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(7), obytes);
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
                noc_async_write_page(row * HT + DS + t, os, l1);
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
        did[1] = qb();
        t[2] = wall();
        did[2] = ql();
        t[3] = wall();
        did[3] = qu();
        t[4] = wall();
        did[4] = qh();
        t[5] = wall();
        did[5] = qhl();
        t[6] = wall();
        did[6] = qy();
        t[7] = wall();
        did[7] = false;
        t[8] = t[7];
        bool any = false;
        for (uint32_t i = 0; i < 8; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        if (!any && w_n >= nexp_g && w_freed_sent >= nexp_g && b_s >= total_rows && x_cred >= total_rows &&
            (!OWN || (l_q >= total_rows && l_cons >= total_rows)) && uq >= total_rows && h_q >= total_rows &&
            (!leads || lq >= total_rows) && drained >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t i = 0; i < 8; ++i) {
        st[470 + i] = tq[i];
    }
}
