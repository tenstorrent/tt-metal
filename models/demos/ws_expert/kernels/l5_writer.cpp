// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups (ws_lane5.py), group BRISC (NOC0). Core (C, Y), j = 5 C + Y.
// Loads this group's slot table (a DRAM page per group, layout in wl_reader.cpp) and the counts, then a non-blocking
// loop over:
//   Q1 x gather / QX x rows to the row's 5 cores / QP partials to the owners at and below / QC landing credits: as
//      g5_writer.cpp, the rows in the chip's compute order (order entry n: expert | slot << 16).
//   Q2 (owners): h stage A - h tile of row q -> slot q % NS of the 5 cores of column C at index 4 C + Y (+ its flag
//      word q + 1), once each of them consumed row q - NS (HCRED words 0-4 by row); their NCRISCs forward the
//      column's 4 tiles along their rows (stage B, l5_reader.cpp).
//   QO own experts: the slots' occupants of kind own, in each slot's order, once the NCRISC freed the slot's previous
//      occupant (FREEW + 16 s >= k): the core's slice from DRAM (PK chunks of CHT tiles, one bank a chunk; chunk q
//      = own index x PKE + CH0 + u), arrival word PK k + chunks landed.
// CT: 0 KSL, 1 DCOLS, 2 OWN, 3 NXR, 4 xring, 5 xflags, 6 xcred, 7 x tile, 8 D, 9 land, 10 lflags, 11 lcred,
//     12 p tile, 13 scratch (16 B words: 0 x flag src, 1 partial flag src, 3 NCRISC's, 4 PAVAIL, 5 PDONE, 6 h flag
//     src), 14 KTD, 15 NS, 16 hring, 17 hflags, 18 h tile, 19 counts L1, 20 table L1, 21 table page bytes, 22 RMAX,
//     23 x row bytes, 24 idx L1, 25 NEXP, 26 weight ring, 27 warr, 28 FREEW, 29 CHT, 30 W tile bytes, 31 RD (own
//     chunk reads in flight), 32 HCRED, 33 K, 34 PKE, then TensorAccessorArgs(x), (idx), (counts), (table), (own)
// RT: 0 j, 1 g, 2 order length, 3 x addr, 4 idx addr, 5 counts addr, 6 stamps, 7 K0, 8 table addr, 9 own slices
//     addr, 10 CH0, 11-15 column NOC x, 16-20 row NOC y
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t p_cb = tt::CBIndex::c_18;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0), DCOLS = get_compile_time_arg_val(1);
    constexpr bool OWN = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t NXR = get_compile_time_arg_val(3), xring = get_compile_time_arg_val(4);
    constexpr uint32_t xflags = get_compile_time_arg_val(5), xcred = get_compile_time_arg_val(6);
    constexpr uint32_t xtile = get_compile_time_arg_val(7), D = get_compile_time_arg_val(8);
    constexpr uint32_t land = get_compile_time_arg_val(9), lflags = get_compile_time_arg_val(10);
    constexpr uint32_t lcred = get_compile_time_arg_val(11), ptile = get_compile_time_arg_val(12);
    constexpr uint32_t scr = get_compile_time_arg_val(13), KTD = get_compile_time_arg_val(14);
    constexpr uint32_t NS = get_compile_time_arg_val(15);
    constexpr uint32_t hring = get_compile_time_arg_val(16), hflags = get_compile_time_arg_val(17);
    constexpr uint32_t htile = get_compile_time_arg_val(18), counts_l1 = get_compile_time_arg_val(19);
    constexpr uint32_t table_l1 = get_compile_time_arg_val(20), TB = get_compile_time_arg_val(21);
    constexpr uint32_t RMAX = get_compile_time_arg_val(22), xrow = get_compile_time_arg_val(23);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(24), NEXP = get_compile_time_arg_val(25);
    constexpr uint32_t wring = get_compile_time_arg_val(26), warr_addr = get_compile_time_arg_val(27);
    constexpr uint32_t freew_addr = get_compile_time_arg_val(28), CHT = get_compile_time_arg_val(29);
    constexpr uint32_t wtile = get_compile_time_arg_val(30), RD = get_compile_time_arg_val(31);
    constexpr uint32_t hcred = get_compile_time_arg_val(32), K = get_compile_time_arg_val(33);
    constexpr uint32_t PKE = get_compile_time_arg_val(34);
    constexpr auto xargs = TensorAccessorArgs<35>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr auto targs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto wargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR, NH = NOWN * NCOL, GC = NR * NCOL;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;
    static_assert(RD >= 1 && RD <= 8);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t xaddr = get_arg_val<uint32_t>(3), iaddr = get_arg_val<uint32_t>(4), caddr = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    const uint32_t K0 = get_arg_val<uint32_t>(7), CH0 = get_arg_val<uint32_t>(10);
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(11 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(16 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);
    const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(8), TB);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(9), wtile);
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    volatile tt_l1_ptr uint32_t* tab = W(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    noc_async_read_page(0, cs, counts_l1 + 64);
    noc_async_read_page(g, ts, table_l1);
    noc_async_read_barrier();
    counts[0] = 1;
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
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
            ++g_r;
            ++g_q;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, KSL)) {
            return false;
        }
        noc_async_read(is.get_noc_addr(order[g_n] & 0xFFFF) + g_r * 32 * 4, idx_l1, 32 * 4);
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
            const uint32_t tc = (C + t) % NCOL;
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
            *W(scr + 16) = pq + 1;
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
        if (*W(scr + 5 * 16) < pq + 1) {  // PDONE
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

    // ---- Q2 (owners): h stage A, h tile -> the 5 cores of the column, once each consumed row q - NS
    uint32_t sent = 0;
    const uint32_t hidx = PC * C + Y;
    auto q2 = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        if (sent >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
        if (sent >= NS) {
            invalidate_l1_cache();
            for (uint32_t k = 0; k < NR; ++k) {
                if (*W(hcred + k * 16) < sent - NS + 1) {
                    return false;
                }
            }
        }
        const uint32_t dst = hring + ((sent % NS) * KTD + hidx) * htile;
        const uint32_t fl = hflags + ((sent % NS) * NH + hidx) * 16;
        *W(scr + 6 * 16) = sent + 1;
        const uint32_t src = get_read_ptr(h_cb);
        for (uint32_t r = 0; r < NR; ++r) {
            noc_async_write(src, get_noc_addr(cx[C], ry[r], dst), htile);
            noc_async_write(scr + 6 * 16, get_noc_addr(cx[C], ry[r], fl), 4);
        }
        noc_async_writes_flushed();
        cb_pop_front(h_cb, 1);
        ++sent;
        return true;
    };

    // ---- QO: own experts' weights from DRAM into their slots (chunk reads with transaction ids 1..RD, in order)
    volatile tt_l1_ptr uint32_t* warr = W(warr_addr);
    volatile tt_l1_ptr uint32_t* freew = W(freew_addr);
    auto entry = [&](uint32_t s, uint32_t k) { return tab + tab[1 + K + s] + 4 * k; };
    uint32_t o_s = 0, o_k[K] = {}, o_issued = 0, o_landed = 0, n_own = 0;
    bool o_busy = false;
    auto next_own = [&]() {
        invalidate_l1_cache();
        for (uint32_t t = 0; t < K; ++t) {
            const uint32_t s = (o_s + t) % K;
            while (o_k[s] < tab[1 + s] && ((entry(s, o_k[s])[0] >> 16) & 3) != 1) {
                ++o_k[s];
            }
            if (o_k[s] < tab[1 + s] && freew[s * 4] >= o_k[s]) {
                o_s = s;
                return true;
            }
        }
        return false;
    };
    auto qo = [&]() -> bool {
        bool did = false;
        if (!o_busy) {
            if (!next_own()) {
                return false;
            }
            o_busy = true;
            o_issued = o_landed = 0;
        }
        const uint32_t s = o_s, k = o_k[s];
        if (o_landed < o_issued) {
            const uint32_t trid = 1 + o_landed % RD;
            if (ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                ++o_landed;
                warr[s * 4] = PK * k + o_landed;
                did = true;
            }
        }
        if (o_issued < PK && o_issued - o_landed < RD) {
            const uint32_t u = o_issued;
            const uint32_t q = entry(s, k)[3] * PKE + CH0 + u;  // chunk q of the chip's own slices
            const uint32_t dst = u < GU_PK ? wring + (s * WGU + u * CHT) * wtile
                                           : wring + (K * WGU + s * WDN + (u - GU_PK) * CHT) * wtile;
            noc_async_read_set_trid(1 + u % RD);
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), dst, CHT * wtile);
            noc_async_read_set_trid(0);
            ++o_issued;
            did = true;
        }
        if (o_landed >= PK) {
            o_busy = false;
            ++o_k[s];
            ++n_own;
            o_s = (s + 1) % K;
        }
        return did;
    };
    auto qo_done = [&]() {
        for (uint32_t s = 0; s < K; ++s) {
            if (o_k[s] < tab[1 + s]) {
                return false;
            }
        }
        return !o_busy;
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint32_t t_start = wall();
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
        did[5] = qo();
        t[6] = wall();
        bool any = false;
        for (uint32_t i = 0; i < 6; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        if (!any && g_n >= nexp_g && g_state == 0 && xq >= total_rows && pq >= total_rows &&
            (!OWN || (cr_sent >= total_rows && sent >= total_rows)) && qo_done()) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t i = 0; i < 6; ++i) {
        st[50 + i] = tq[i];
    }
    st[56] = n_own;
    st[495] = wall() - t_start;
    st[496] = t_start;
}
