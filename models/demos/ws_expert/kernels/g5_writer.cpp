// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert 5 x 5 K-split groups, BRISC (NOC0). Core (C, Y) of a group, j = 5 C + Y. A non-blocking loop over:
//   Q1 x gather: rows q with q % 5 == C of the group's experts: the row tile's 32 token ids, then 32 reads of KSL K
//      tiles (K0 ..) into CB 7, row-major (compute tilizes them into CB 8).
//   QX: tilized row q (CB 8) -> slot q % NXR of the x ring of the 5 cores of row Y (payload, then flag q + 1), once
//      all 5 consumed row q - NXR (XCRED word of each consumer column).
//   QP partials to the owners o >= Y (rows below and itself; o < 4): tiles 2 o, 2 o + 1 of row q (CB 18) -> o's
//      landing slot q % D at contributor Y (+ flag q + 1), once o consumed row q - D (LCRED word o). PAVAIL = q + 1
//      for the NCRISC (owners above); the row is popped once the NCRISC's PDONE reaches q + 1.
//   QC (owners): this core's consumed landing rows (LCRED word Y, kept by the NCRISC) to the rows below.
//   Q2 (owners): h tile of row q -> the column leader (C, q % 4): its h ring slot, then its HARR semaphore.
//   Q3 (owners, even columns): leader of rows q = Y + 4 k: once 4 h tiles arrived, multicast the column's 4 h tiles
//      to the group (+ flag word (slot, C)); odd columns' leaders run on the NCRISC (NOC1).
// CT: 0 KSL, 1 DCOLS, 2 OWN, 3 NXR, 4 xring, 5 xflags, 6 xcred, 7 x tile, 8 D, 9 land, 10 lflags, 11 lcred,
//     12 p tile, 13 scratch (16 B words: 0 x flag src, 1 BRISC partial flag src, 3 NCRISC's, 4 PAVAIL, 5 PDONE,
//     6 BRISC h flag src, 7 NCRISC's), 14 KTD, 15 G, 16 NS, 17 hring, 18 hflags, 19 h tile, 20 counts L1,
//     21-24 group rect (NOC0), 25 RMAX, 26 x row bytes, 27 idx L1, 28 NEXP, then TensorAccessorArgs(x), (idx), (counts)
// RT: 0 j, 1 g, 2 nexp_g, 3 x addr, 4 idx addr, 5 counts addr, 6 stamps, 7 K0, 8-12 column NOC x, 13-17 row NOC y
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t p_cb = tt::CBIndex::c_18;
constexpr uint32_t SEM_HARR = 1;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0);
    constexpr bool OWN = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t NXR = get_compile_time_arg_val(3), xring = get_compile_time_arg_val(4);
    constexpr uint32_t xflags = get_compile_time_arg_val(5), xcred = get_compile_time_arg_val(6);
    constexpr uint32_t xtile = get_compile_time_arg_val(7), D = get_compile_time_arg_val(8);
    constexpr uint32_t land = get_compile_time_arg_val(9), lflags = get_compile_time_arg_val(10);
    constexpr uint32_t lcred = get_compile_time_arg_val(11), ptile = get_compile_time_arg_val(12);
    constexpr uint32_t scr = get_compile_time_arg_val(13), KTD = get_compile_time_arg_val(14);
    constexpr uint32_t G = get_compile_time_arg_val(15), NS = get_compile_time_arg_val(16);
    constexpr uint32_t hring = get_compile_time_arg_val(17), hflags = get_compile_time_arg_val(18);
    constexpr uint32_t htile = get_compile_time_arg_val(19), counts_l1 = get_compile_time_arg_val(20);
    constexpr uint32_t gx0 = get_compile_time_arg_val(21), gy0 = get_compile_time_arg_val(22);
    constexpr uint32_t gx1 = get_compile_time_arg_val(23), gy1 = get_compile_time_arg_val(24);
    constexpr uint32_t RMAX = get_compile_time_arg_val(25), xrow = get_compile_time_arg_val(26);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(27), NEXP = get_compile_time_arg_val(28);
    constexpr auto xargs = TensorAccessorArgs<29>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR;
    constexpr uint32_t ngroup = (gx1 - gx0 + 1) * (gy1 - gy0 + 1);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t xaddr = get_arg_val<uint32_t>(3), iaddr = get_arg_val<uint32_t>(4), caddr = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    const uint32_t K0 = get_arg_val<uint32_t>(7);
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(8 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(13 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
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
    const uint32_t koff = K0 * 64;
    uint32_t g_n = 0, g_r = 0, g_q = 0, g_state = 0;
    auto q1 = [&]() -> bool {
        if (g_state == 1) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* idx = W(idx_l1);
            cb_reserve_back(xrm_cb, KSL);
            uint32_t l1 = get_write_ptr(xrm_cb);
            for (uint32_t i = 0; i < 32; ++i, l1 += KSL * 64) {
                noc_async_read(xs.get_noc_addr(idx[i]) + koff, l1, KSL * 64);
            }
            g_state = 2;
            return true;
        }
        if (g_state == 2) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            cb_push_back(xrm_cb, KSL);
            ++g_r;
            ++g_q;
            g_state = 0;
            return true;
        }
        while (true) {
            while (g_n < nexp_g && g_r >= rows_of(g_n)) {
                ++g_n;
                g_r = 0;
            }
            if (g_n >= nexp_g || g_q % NCOL == C) {
                break;
            }
            ++g_r;  // (another column's row)
            ++g_q;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, KSL)) {
            return false;
        }
        noc_async_read(is.get_noc_addr(g + g_n * G) + g_r * 32 * 4, idx_l1, 32 * 4);
        g_state = 1;
        return true;
    };

    // ---- QX: own rows to the 5 cores of the row
    uint32_t xq = C;
    auto qx = [&]() -> bool {
        if (xq >= total_rows || !cb_pages_available_at_front(xst_cb, KSL)) {
            return false;
        }
        if (xq >= NXR) {
            invalidate_l1_cache();
            for (uint32_t t = 0; t < NCOL; ++t) {
                if (*W(xcred + t * 16) < xq - NXR + 1) {
                    return false;
                }
            }
        }
        const uint32_t dst = xring + (xq % NXR) * KSL * xtile, fl = xflags + (xq % NXR) * 16;
        *W(scr) = xq + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        for (uint32_t t = 0; t < NCOL; ++t) {
            const uint32_t tc = (C + t) % NCOL;  // (itself first, then eastwards)
            noc_async_write(src, get_noc_addr(cx[tc], ry[Y], dst), KSL * xtile);
            noc_async_write(scr, get_noc_addr(cx[tc], ry[Y], fl), 4);
        }
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, KSL);
        xq += NCOL;
        return true;
    };

    // ---- QP: partials to the owners at and below this row
    uint32_t pq = 0, po = Y;
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
        while (po < NOWN) {
            if (pq >= D) {
                invalidate_l1_cache();
                if (*W(lcred + po * 16) < pq - D + 1) {
                    return did;
                }
            }
            noc_async_write(
                src + 2 * po * ptile, get_noc_addr(cx[C], ry[po], land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 16, get_noc_addr(cx[C], ry[po], lflags + (s * NR + Y) * 16), 4);
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

    // ---- QC (owners): landing credits to the rows below
    uint32_t cr_sent = 0;
    auto qc = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        invalidate_l1_cache();
        const uint32_t c = *W(lcred + Y * 16);
        if (c <= cr_sent) {
            return false;
        }
        for (uint32_t o = Y + 1; o < NR; ++o) {
            noc_inline_dw_write(get_noc_addr(cx[C], ry[o], lcred + Y * 16), c);
        }
        cr_sent = c;
        return true;
    };

    // ---- Q2 (owners): h tile -> column leader
    uint32_t sent = 0;
    const uint32_t harr = get_semaphore(SEM_HARR);
    auto q2 = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        if (sent >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
        const uint32_t ly = ry[sent % NOWN];
        noc_async_write(
            get_read_ptr(h_cb), get_noc_addr(cx[C], ly, hring + ((sent % NS) * KTD + PC * C + Y) * htile), htile);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx[C], ly, harr), 1);
        cb_pop_front(h_cb, 1);
        ++sent;
        return true;
    };
    // ---- Q3 (owners, even columns): leader multicasts
    const bool leads = OWN && C % 2 == 0;
    uint32_t led = 0, lq = Y;
    auto q3 = [&]() -> bool {
        if (!leads || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(harr) < NOWN * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + PC * C) * htile;
        *W(scr + 6 * 16) = lq + 1;
        noc_async_write_multicast_loopback_src(
            part, get_noc_multicast_addr(gx0, gy0, gx1, gy1, part), PC * htile, ngroup, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr + 6 * 16, get_noc_multicast_addr(gx0, gy0, gx1, gy1, hflags + ((lq % NS) * NCOL + C) * 16), ngroup);
        noc_async_writes_flushed();
        ++led;
        lq += NOWN;
        return true;
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t tq[6] = {0, 0, 0, 0, 0, 0};
    while (true) {
        bool did[6];
        uint32_t t[7];
        t[0] = wall();
        did[0] = qx();
        t[1] = wall();
        did[1] = q1();
        t[2] = wall();
        did[2] = qp();
        t[3] = wall();
        did[3] = qc();
        t[4] = wall();
        did[4] = q2();
        t[5] = wall();
        did[5] = q3();
        t[6] = wall();
        bool any = false;
        for (uint32_t i = 0; i < 6; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        if (!any && g_n >= nexp_g && g_state == 0 && xq >= total_rows && pq >= total_rows &&
            (!OWN || (cr_sent >= total_rows && sent >= total_rows)) && (!leads || lq >= total_rows)) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t i = 0; i < 6; ++i) {
        st[50 + i] = tq[i];
    }
}
