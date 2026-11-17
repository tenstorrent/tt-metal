// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups, h by the group hub's multicast with the NoC split by direction
// (ws_lane5.py, L5Cfg.hmc 2): group BRISC (NOC0). Only transfers that stay inside the group's rectangle on NOC0 (east /
// south of this core) go out from here, so nothing enters a group's rectangle on NOC0 from outside; the NCRISC
// (l5h_reader.cpp, WL5_HMC2) carries the DRAM reads, the fabric, the hub's tiles and everything west / north. Loop:
//   QX: own x rows to the cores of row Y at and east of this column; the row is published (XAVAIL / XADDR) for the
//      NCRISC's westward copies and popped once they are sent (XDONE).
//   QP: partials to the owners at and below this row (PAVAIL / PDONE with the NCRISC's owners above), as l5_writer.cpp.
//   QC (owners): landing credits to the rows below, as l5_writer.cpp.
//   QM (hub, core 0, the rectangle's NOC0 multicast start corner): row q once its 20 h tiles are in slot q % NS of
//      this core's h ring (HFLAGS) and every core of the group consumed row q - NS (HCONS): the slot to the same slot
//      of the 25 cores, then its flag (HCOLF word 5 s = q + 1) through the same command buffer; rows sent -> scratch
//      word 3.
// The counts and the slot table are loaded by the NCRISC (counts word 0 set once they are in). WL5_GATHER0 (diagnostic,
// outside the NoC rule: DRAM read responses enter the rectangle on NOC0): the x gather from here, as l5_writer.cpp.
//   QT (WL5_YT): the down row tiles (CB 17) transposed into row-major rows in a staging slot for the NCRISC's row
//      writes (the compute packs plain tiles: no untilize in the packer).
// CT: as l5_writer.cpp (0-34), then 35 HCONS, 36 HCOLF (37 the staging slot with WL5_YT); RT: as l5_writer.cpp (0-20), then 21-24 the group's NOC0
//     rectangle (x0, y0, x1, y1). Scratch words (16 B): 0 x flag, 1 partial flag, 2 hub flag, 3 rows multicast,
//     4 PAVAIL, 5 PDONE, 8 XAVAIL, 9 XADDR, 10 XDONE.
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
    constexpr uint32_t hcons = get_compile_time_arg_val(wargs.next_compile_time_args_offset());
    constexpr uint32_t hcolf = get_compile_time_arg_val(wargs.next_compile_time_args_offset() + 1);
#if defined(WL5_YT)
    constexpr uint32_t ystage = get_compile_time_arg_val(wargs.next_compile_time_args_offset() + 2);
#endif
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR, NH = NOWN * NCOL, GC = NR * NCOL;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;
    static_assert(RD >= 1 && RD <= 8);

    const uint32_t J = get_arg_val<uint32_t>(0), nexp_g = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
#if defined(WL5_GATHER0)
    const auto xs = TensorAccessor(xargs, get_arg_val<uint32_t>(3), xrow);
    const auto is = TensorAccessor(iargs, get_arg_val<uint32_t>(4), RMAX * 32 * 4);
    const uint32_t K0 = get_arg_val<uint32_t>(7);
#endif
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(11 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(16 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    volatile tt_l1_ptr uint32_t* tab = W(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

#if defined(WL5_GATHER0)
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

#endif
    // ---- QX: own rows to the cores of the row at and east of this column (the NCRISC sends the rest westwards)
    uint32_t xq = C;
    bool x_pub = false;
    auto qx = [&]() -> bool {
        if (xq >= total_rows || !cb_pages_available_at_front(xst_cb, KSL)) {
            return false;
        }
        const uint32_t src = get_read_ptr(xst_cb);
        if (!x_pub) {
            if (xq >= NXR) {
                invalidate_l1_cache();
                for (uint32_t t = C; t < NCOL; ++t) {
                    if (*W(xcred + t * 16) < xq - NXR + 1) {
                        return false;
                    }
                }
            }
            const uint32_t dst = xring + (xq % NXR) * KSL * xtile, fl = xflags + (xq % NXR) * 16;
            *W(scr) = xq + 1;
            for (uint32_t tc = C; tc < NCOL; ++tc) {
                noc_async_write(src, get_noc_addr(cx[tc], ry[Y], dst), KSL * xtile);
                noc_async_write(scr, get_noc_addr(cx[tc], ry[Y], fl), 4);
            }
            *W(scr + 9 * 16) = src;
            *W(scr + 8 * 16) = xq + 1;  // XAVAIL
            x_pub = true;
            return true;
        }
        invalidate_l1_cache();
        if (*W(scr + 10 * 16) < xq + 1) {  // XDONE
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, KSL);
        xq += NCOL;
        x_pub = false;
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

#if defined(WL5_YT)
    // ---- QT: the compute's down row tiles (CB 17, DCOLS bf16 tiles) to row-major rows in the staging slot (row r at
    // r DCOLS 64 B: tile c's columns at c 64), once the NCRISC has written out the previous row tile (YDONE, scratch
    // word 13); then YREADY (word 12) = rows staged. A tile's row r is its face row r of faces 0 / 1 (r < 16) or 2 / 3.
    constexpr uint32_t y_cb = tt::CBIndex::c_17;
    constexpr uint32_t seg = DCOLS * 64;
    uint32_t t_q = 0;
    auto qt = [&]() -> bool {
        if (t_q >= total_rows || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        invalidate_l1_cache();
        if (t_q > 0 && *W(scr + 13 * 16) < t_q) {
            return false;
        }
        const uint32_t src = get_read_ptr(y_cb);
        for (uint32_t c = 0; c < DCOLS; ++c) {
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(src + c * 2048);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t* f = tile + ((r < 16 ? 0 : 256) + (r & 15) * 8);  // (face 0 / 2 row; face 1 / 3 at + 128)
                uint32_t* d = reinterpret_cast<uint32_t*>(ystage + r * seg + c * 64);
#pragma GCC unroll 8
                for (uint32_t i = 0; i < 8; ++i) {
                    d[i] = f[i];
                    d[8 + i] = f[128 + i];
                }
            }
        }
        asm volatile("fence" ::: "memory");
        *W(scr + 12 * 16) = t_q + 1;  // YREADY
        cb_pop_front(y_cb, DCOLS);
        ++t_q;
        return true;
    };
#endif
    // ---- QM (hub): h rows multicast to the group
    const uint32_t mx0 = get_arg_val<uint32_t>(21), my0 = get_arg_val<uint32_t>(22);
    const uint32_t mx1 = get_arg_val<uint32_t>(23), my1 = get_arg_val<uint32_t>(24);
    uint32_t m_q = 0;
    auto qm = [&]() -> bool {
        if (J != 0 || m_q >= total_rows) {
            return false;
        }
        const uint32_t s = m_q % NS;
        invalidate_l1_cache();
        for (uint32_t i = 0; i < NH; ++i) {
            if (*W(hflags + (s * NH + i) * 16) < m_q + 1) {
                return false;
            }
        }
        if (m_q >= NS) {
            for (uint32_t k = 0; k < GC; ++k) {
                if (*W(hcons + k * 16) < m_q - NS + 1) {
                    return false;
                }
            }
        }
        *W(scr + 2 * 16) = m_q + 1;
        const uint32_t slot = hring + s * KTD * htile;
        noc_async_write_multicast_loopback_src(
            slot, get_noc_multicast_addr(mx0, my0, mx1, my1, slot), KTD * htile, GC, false);
        noc_async_write_multicast_loopback_src(
            scr + 2 * 16, get_noc_multicast_addr(mx0, my0, mx1, my1, hcolf + s * NCOL * 16), 4, GC, false);
        noc_async_writes_flushed();
        *W(scr + 3 * 16) = m_q + 1;
        ++m_q;
        return true;
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
#if defined(WL5_GATHER0)
        did[1] = q1();
        t[2] = wall();
#else
        did[1] = false;
        t[2] = t[1];
#endif
        did[2] = qp();
        t[3] = wall();
        did[3] = qc();
        t[4] = wall();
        did[4] = qm();
        t[5] = wall();
#if defined(WL5_YT)
        did[5] = qt();
        t[6] = wall();
#else
        did[5] = false;
        t[6] = t[5];
#endif
        bool any = false;
        for (uint32_t i = 0; i < 6; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
#if defined(WL5_GATHER0)
        const bool gathered = g_n >= nexp_g && g_state == 0;
#else
        const bool gathered = true;
#endif
#if defined(WL5_YT)
        const bool staged = t_q >= total_rows;
#else
        const bool staged = true;
#endif
        if (!any && gathered && staged && xq >= total_rows && pq >= total_rows && (!OWN || cr_sent >= total_rows) &&
            (J != 0 || m_q >= total_rows)) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t i = 0; i < 6; ++i) {
        st[50 + i] = tq[i];
    }
    st[495] = wall() - t_start;
    st[496] = t_start;
}
