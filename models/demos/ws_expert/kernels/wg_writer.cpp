// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert groups BRISC (NOC0): loads the expert counts into L1 (word 0 = ready), then a non-blocking loop over:
//   Q1 x gather: this core's x blocks (block l of an expert with l % GC == j; row l / BPR, K block l % BPR): the row
//      tile's 32 token ids (128 B of the expert's index table), then 32 rows x BK K-tiles into CB 7, row-major.
//   Q2 h out: each row's h tile (CB 16) -> its column leader's h ring slot (row q's leader in column c is core
//      (c, q % 10)) at this core's offset, then the leader's HARR += 1.
//   Q3 leader: for the rows this core leads, once HARR counts all 10 of its column, multicast the column's 10 h tiles
//      (linked) and its flag word (q + 1) to the whole group: 2 multicasts a row instead of 20.
//   Q4 down outputs (CB 17): the group's last expert's rows to DRAM (out rows of group g at g * RMAX * 32).
// RT: 0 j, 1 g, 2 nexp_g, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 out addr, 7 x addr, 8 idx addr,
//     9 counts addr
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 GC, 5 G, 6 NS, 7 hring, 8 hflags, 9-12 group rect NOC0, 13 h tile bytes, 14 HT,
//     15 RMAX, 16 x row bytes, 17 counts L1, 18 idx scratch L1, 19 NEXP, then TensorAccessorArgs(out), (x), (idx),
//     (counts)
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t SEM_HARR = 1, SEM_SCR = 2;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t BK = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t GC = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t NS = get_compile_time_arg_val(6);
    constexpr uint32_t hring = get_compile_time_arg_val(7);
    constexpr uint32_t hflags = get_compile_time_arg_val(8);
    constexpr uint32_t gx0 = get_compile_time_arg_val(9), gy0 = get_compile_time_arg_val(10);
    constexpr uint32_t gx1 = get_compile_time_arg_val(11), gy1 = get_compile_time_arg_val(12);
    constexpr uint32_t htile = get_compile_time_arg_val(13);
    constexpr uint32_t HT = get_compile_time_arg_val(14);
    constexpr uint32_t RMAX = get_compile_time_arg_val(15);
    constexpr uint32_t xrow = get_compile_time_arg_val(16);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(17);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(18);
    constexpr uint32_t NEXP = get_compile_time_arg_val(19);
    constexpr auto oargs = TensorAccessorArgs<20>();
    constexpr auto xargs = TensorAccessorArgs<oargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr uint32_t BPR = KT / BK;
    constexpr uint32_t HALF = GC / 2;  // cores a column

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint32_t oaddr = get_arg_val<uint32_t>(6), xaddr = get_arg_val<uint32_t>(7);
    const uint32_t iaddr = get_arg_val<uint32_t>(8), caddr = get_arg_val<uint32_t>(9);
    const uint32_t C = J / HALF, Y = J % HALF;
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    noc_async_read_page(0, cs, counts_l1 + 64);
    noc_async_read_barrier();
    counts[0] = 1;
    auto rows_of = [&](uint32_t n) { return (counts[16 + g + n * G] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

    // ---- Q1
    uint32_t g_n = 0, g_l = J;
    auto q1 = [&]() -> bool {
        while (g_n < nexp_g && g_l >= rows_of(g_n) * BPR) {
            ++g_n;
            g_l = J;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, BK)) {
            return false;
        }
        const uint32_t r = g_l / BPR, k = (g_l % BPR) * BK;
#ifdef WG_NO_GATHER
        cb_reserve_back(xrm_cb, BK);
        cb_push_back(xrm_cb, BK);
        g_l += GC;
        return true;
#endif
        noc_async_read(is.get_noc_addr(g + g_n * G) + r * 32 * 4, idx_l1, 32 * 4);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
        cb_reserve_back(xrm_cb, BK);
        uint32_t l1 = get_write_ptr(xrm_cb);
        for (uint32_t i = 0; i < 32; ++i, l1 += BK * 64) {
            noc_async_read(xs.get_noc_addr(idx[i]) + k * 64, l1, BK * 64);
        }
        noc_async_read_barrier();
        cb_push_back(xrm_cb, BK);
        g_l += GC;
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
    // ---- Q3: leader multicasts (rows q with q % 10 == Y)
    uint32_t led = 0, lq = Y;
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    constexpr uint32_t ngroup = (gx1 - gx0 + 1) * (gy1 - gy0 + 1);
    auto q3 = [&]() -> bool {
        if (lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(harr) < HALF * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + C * HALF) * htile;
        *scr = lq + 1;
        noc_async_write_multicast_loopback_src(
            part, get_noc_multicast_addr(gx0, gy0, gx1, gy1, part), HALF * htile, ngroup, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr_addr, get_noc_multicast_addr(gx0, gy0, gx1, gy1, hflags + ((lq % NS) * 2 + C) * 16), ngroup);
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };
    // ---- Q4: down outputs
    uint32_t drained = 0, d_n = 0, d_r = 0;
    auto q4 = [&]() -> bool {
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
    while (true) {
        const bool a = q1();
        const bool b = q2();
        const bool c = q3();
        const bool d = q4();
        if (!a && !b && !c && !d && g_n >= nexp_g && sent >= total_rows && lq >= total_rows &&
            drained >= total_rows) {
            break;
        }
    }
    noc_async_atomic_barrier();
}
