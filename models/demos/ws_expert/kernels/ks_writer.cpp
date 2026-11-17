// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert K-split groups BRISC (NOC0). Core (C, Y) of a group (j = 10 C + Y). A non-blocking loop over:
//   Q1 x gather: for every row of the group's experts, the row tile's 32 token ids, then 32 rows x KH K-tiles (K tiles
//      [KSL Y + KH C, +KH)) into CB 7, row-major (compute tilizes them into CB 8).
//   QX (column 0; column 1's run on the NCRISC, NOC1 west): this core's tilized half of x row s (CB 8) -> half C of
//      slot s % NXR of its own and its partner's x ring (payload + flag word s + 1 from one command buffer), once
//      both consumed row s - NXR (XCRED words 0 / 1).
//   QP partials, rows below (NOC0) and itself: row q's 20 partial tiles (CB 18); contribution to owner o (pair
//      10 C + o: partial tiles 2 o, 2 o + 1) -> o's landing slot q % D at contributor index Y (+ flag q + 1), once o
//      consumed row q - D (LCRED word o). Publishes PAVAIL = q + 1 for the NCRISC (rows above), pops the row once
//      the NCRISC's PDONE reaches q + 1.
//   QC landing credits to the rows below: this core's consumed landing rows (LCRED word Y, kept by the NCRISC).
//   Q2 h out / Q3 column 0 h leader multicasts: as the 1 x 20 groups (core j owns h pair j).
// RT: 0 j, 1 g, 2 nexp_g, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 x addr, 7 idx addr, 8 counts addr,
//     9 stamps
// CT: 0 KSL, 1 KTD, 2 GC, 3 G, 4 NS, 5 hring, 6 hflags, 7-10 group rect NOC0, 11 h tile, 12 RMAX, 13 x row bytes,
//     14 counts L1, 15 idx L1, 16 NEXP, 17 NXR, 18 xring, 19 xflags, 20 xcred, 21 x tile, 22 D, 23 land, 24 lflags,
//     25 lcred, 26 p tile, 27 scratch (16 B words: 0 BRISC x flag src, 1 BRISC partial flag src, 2/3 NCRISC's,
//     4 PAVAIL, 5 PDONE), then TensorAccessorArgs(x), (idx), (counts)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t p_cb = tt::CBIndex::c_18;
constexpr uint32_t SEM_HARR = 1, SEM_SCR = 2;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0), KTD = get_compile_time_arg_val(1);
    constexpr uint32_t GC = get_compile_time_arg_val(2), G = get_compile_time_arg_val(3);
    constexpr uint32_t NS = get_compile_time_arg_val(4);
    constexpr uint32_t hring = get_compile_time_arg_val(5), hflags = get_compile_time_arg_val(6);
    constexpr uint32_t gx0 = get_compile_time_arg_val(7), gy0 = get_compile_time_arg_val(8);
    constexpr uint32_t gx1 = get_compile_time_arg_val(9), gy1 = get_compile_time_arg_val(10);
    constexpr uint32_t htile = get_compile_time_arg_val(11), RMAX = get_compile_time_arg_val(12);
    constexpr uint32_t xrow = get_compile_time_arg_val(13), counts_l1 = get_compile_time_arg_val(14);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(15), NEXP = get_compile_time_arg_val(16);
    constexpr uint32_t NXR = get_compile_time_arg_val(17), xring = get_compile_time_arg_val(18);
    constexpr uint32_t xflags = get_compile_time_arg_val(19), xcred = get_compile_time_arg_val(20);
    constexpr uint32_t xtile = get_compile_time_arg_val(21), D = get_compile_time_arg_val(22);
    constexpr uint32_t land = get_compile_time_arg_val(23), lflags = get_compile_time_arg_val(24);
    constexpr uint32_t lcred = get_compile_time_arg_val(25), ptile = get_compile_time_arg_val(26);
    constexpr uint32_t scr = get_compile_time_arg_val(27);
    constexpr auto xargs = TensorAccessorArgs<28>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr uint32_t HALF = GC / 2, KH = KSL / 2, NW = 2 * HALF, LT = 2 * HALF;
#ifdef KS_XALT
    constexpr uint32_t XP = KSL, XSTEP = 2;  // whole rows q with q % 2 == C
#else
    constexpr uint32_t XP = KH, XSTEP = 1;  // half of every row
#endif

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint32_t xaddr = get_arg_val<uint32_t>(6), iaddr = get_arg_val<uint32_t>(7), caddr = get_arg_val<uint32_t>(8);
    const uint32_t C = J / HALF, Y = J % HALF;
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    noc_async_read_page(0, cs, counts_l1 + 64);
    noc_async_read_barrier();
    counts[0] = 1;
    auto rows_of = [&](uint32_t n) { return (counts[16 + g + n * G] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

    // ---- Q1: gather state machine (0 idle, 1 ids requested, 2 rows requested)
#ifdef KS_XALT
    const uint32_t koff = KSL * Y * 64;                           // bytes into a token row
    uint32_t g_n = 0, g_r = 0, g_q = 0, g_state = 0, g_done = 0;  // (g_q: the row's index in the group's stream)
#else
    const uint32_t koff = (KSL * Y + KH * C) * 64;
    uint32_t g_n = 0, g_r = 0, g_q = 0, g_state = 0, g_done = 0;
#endif
    auto q1 = [&]() -> bool {
        if (g_state == 1) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* idx = W(idx_l1);
            cb_reserve_back(xrm_cb, XP);
            uint32_t l1 = get_write_ptr(xrm_cb);
            for (uint32_t i = 0; i < 32; ++i, l1 += XP * 64) {
                noc_async_read(xs.get_noc_addr(idx[i]) + koff, l1, XP * 64);
            }
            g_state = 2;
            return true;
        }
        if (g_state == 2) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            cb_push_back(xrm_cb, XP);
            ++g_r;
            ++g_q;
            ++g_done;
            g_state = 0;
            return true;
        }
        while (true) {
            while (g_n < nexp_g && g_r >= rows_of(g_n)) {
                ++g_n;
                g_r = 0;
            }
            if (g_n >= nexp_g || XSTEP == 1 || g_q % 2 == C) {
                break;
            }
            ++g_r;  // (KS_XALT: the partner's row)
            ++g_q;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, XP)) {
            return false;
        }
        noc_async_read(is.get_noc_addr(g + g_n * G) + g_r * 32 * 4, idx_l1, 32 * 4);
        g_state = 1;
        return true;
    };

    // ---- QX: x half of row s -> own and partner's x ring (column 0 here; column 1 on the NCRISC)
    const uint32_t px = cxs[1 - C], py = ny0 + Y, mx = cxs[C];
    uint32_t xq = XSTEP == 2 ? C : 0;
    auto qx = [&]() -> bool {
        if (C != 0 || xq >= total_rows || !cb_pages_available_at_front(xst_cb, XP)) {
            return false;
        }
        if (xq >= NXR) {
            invalidate_l1_cache();
            if (*W(xcred) < xq - NXR + 1 || *W(xcred + 16) < xq - NXR + 1) {
                return false;
            }
        }
        const uint32_t dst = xring + ((xq % NXR) * KSL + (XSTEP == 2 ? 0 : C * KH)) * xtile;
        const uint32_t fl = xflags + ((xq % NXR) * 2 + C) * 16;
        *W(scr) = xq + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        noc_async_write(src, get_noc_addr(mx, py, dst), XP * xtile);
        noc_async_write(scr, get_noc_addr(mx, py, fl), 4);
        noc_async_write(src, get_noc_addr(px, py, dst), XP * xtile);
        noc_async_write(scr, get_noc_addr(px, py, fl), 4);
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, XP);
        xq += XSTEP;
        return true;
    };

    // ---- QP: partials to the rows below and to itself
    uint32_t pq = 0, po = Y;  // row, next target
    bool p_av = false;
    auto qp = [&]() -> bool {
        if (pq >= total_rows) {
            return false;
        }
        if (!p_av) {
            if (!cb_pages_available_at_front(p_cb, NW)) {
                return false;
            }
            p_av = true;
            *W(scr + 4 * 16) = pq + 1;  // PAVAIL
            *W(scr + 16) = pq + 1;      // this RISC's partial flag source
        }
        bool did = false;
        const uint32_t src = get_read_ptr(p_cb), s = pq % D;
        while (po < HALF) {
            if (pq >= D) {
                invalidate_l1_cache();
                if (*W(lcred + po * 16) < pq - D + 1) {
                    return did;
                }
            }
            const uint32_t oy = ny0 + po;
            noc_async_write(src + 2 * po * ptile, get_noc_addr(mx, oy, land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 16, get_noc_addr(mx, oy, lflags + (s * HALF + Y) * 16), 4);
            ++po;
            did = true;
        }
        invalidate_l1_cache();
        if (*W(scr + 5 * 16) < pq + 1) {  // PDONE: the NCRISC's part
            return did;
        }
        noc_async_writes_flushed();
        cb_pop_front(p_cb, NW);
        ++pq;
        po = Y;
        p_av = false;
        return true;
    };

    // ---- QC: landing credits to the rows below
    uint32_t cr_sent = 0;
    auto qc = [&]() -> bool {
        invalidate_l1_cache();
        const uint32_t c = *W(lcred + Y * 16);
        if (c <= cr_sent) {
            return false;
        }
        for (uint32_t o = Y + 1; o < HALF; ++o) {
            noc_inline_dw_write(get_noc_addr(mx, ny0 + o, lcred + Y * 16), c);
        }
        cr_sent = c;
        return true;
    };

    // ---- Q2: h tile -> column leader
    uint32_t sent = 0;
    const uint32_t harr = get_semaphore(SEM_HARR);
    auto q2 = [&]() -> bool {
        if (sent >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
        const uint32_t lx = cxs[C], ly = ny0 + sent % HALF;
        noc_async_write(
            get_read_ptr(h_cb), get_noc_addr(lx, ly, hring + ((sent % NS) * KTD + C * HALF + Y) * htile), htile);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, harr), 1);
        cb_pop_front(h_cb, 1);
        ++sent;
        return true;
    };
    // ---- Q3: column 0 leader multicasts
    uint32_t led = 0, lq = Y;
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    constexpr uint32_t ngroup = (gx1 - gx0 + 1) * (gy1 - gy0 + 1);
    auto q3 = [&]() -> bool {
        if (C != 0 || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(harr) < HALF * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + C * HALF) * htile;
        *W(scr_addr) = lq + 1;
        noc_async_write_multicast_loopback_src(
            part, get_noc_multicast_addr(gx0, gy0, gx1, gy1, part), HALF * htile, ngroup, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr_addr, get_noc_multicast_addr(gx0, gy0, gx1, gy1, hflags + ((lq % NS) * 2 + C) * 16), ngroup);
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t tq[6] = {0, 0, 0, 0, 0, 0};
    while (true) {
        uint32_t t0 = wall();
        const bool a = qx();
        uint32_t t1 = wall();
        const bool b = q1();
        uint32_t t2 = wall();
        const bool c = qp();
        uint32_t t3 = wall();
        const bool d = qc();
        uint32_t t4 = wall();
        const bool e = q2();
        uint32_t t5 = wall();
        const bool f = q3();
        uint32_t t6 = wall();
        if (a) {
            tq[0] += t1 - t0;
        }
        if (b) {
            tq[1] += t2 - t1;
        }
        if (c) {
            tq[2] += t3 - t2;
        }
        if (d) {
            tq[3] += t4 - t3;
        }
        if (e) {
            tq[4] += t5 - t4;
        }
        if (f) {
            tq[5] += t6 - t5;
        }
        if (!a && !b && !c && !d && !e && !f && g_n >= nexp_g && g_state == 0 && (C != 0 || xq >= total_rows) &&
            pq >= total_rows && cr_sent >= total_rows && sent >= total_rows && (C != 0 || lq >= total_rows)) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* st = W(get_arg_val<uint32_t>(9));
    for (uint32_t i = 0; i < 6; ++i) {
        st[50 + i] = tq[i];
    }
}
