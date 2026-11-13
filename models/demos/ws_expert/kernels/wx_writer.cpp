// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert x-streaming BRISC (NOC0): a non-blocking loop over two in-order queues (a blocking wait on one could hold
// back a block a peer needs from the other):
//   Q1 x gather: this core's x blocks, in stream order: the 32 token rows of row tile r (row ids from the index
//      table; padding rows point at a zero row), K-tiles [k, k + n) of each (n * 64 B a row), into CB 7 as a
//      row-major 32 x BK-tile block (row pitch BK * 64 B) for compute's tilize.
//   Q2 compute outputs in program order: SEND_L / SEND_R partials (to the neighbour's ring, after its grant), h rows
//      (linked multicast + flag to the column's h ring), DN ops' down outputs (the last expert's to DRAM).
// RT: 0 nop, 1 J, 2 left x, 3 left y, 4 out addr, 5 nown (owned rows), 6 right x, 7 right y, 8 x addr, 9 idx addr,
//     10 NB (own x blocks an expert), 11.. own[nown], then per op: flags; then per own block: r, k, n
// CT: 0 NPAIR, 1 NS, 2 KTD, 3 GC, 4 NEXP, 5 hring, 6 hflags, 7 pinL, 8-11 column rect NOC0, 12 h tile bytes,
//     13 DCOLS, 14 HT, 15 pinR, 16 BK, 17 R (row tiles), 18 x row bytes, 19 idx L1 scratch addr, then
//     TensorAccessorArgs(out), (x), (idx)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3, SEM_SCR = 4;
constexpr uint32_t F_SEND_L = 4, F_SEND_R = 8, F_DN = 16, F_TIL = 32 | 64;

void kernel_main() {
    constexpr uint32_t npair = get_compile_time_arg_val(0);
    constexpr uint32_t NS = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t GC = get_compile_time_arg_val(3);
    constexpr uint32_t NEXP = get_compile_time_arg_val(4);
    constexpr uint32_t hring = get_compile_time_arg_val(5);
    constexpr uint32_t hflags = get_compile_time_arg_val(6);
    constexpr uint32_t pinL = get_compile_time_arg_val(7);
    constexpr uint32_t cx0 = get_compile_time_arg_val(8), cy0 = get_compile_time_arg_val(9);
    constexpr uint32_t cx1 = get_compile_time_arg_val(10), cy1 = get_compile_time_arg_val(11);
    constexpr uint32_t htile = get_compile_time_arg_val(12);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(13);
    constexpr uint32_t HT = get_compile_time_arg_val(14);
    constexpr uint32_t pinR = get_compile_time_arg_val(15);
    constexpr uint32_t BK = get_compile_time_arg_val(16);
    constexpr uint32_t R = get_compile_time_arg_val(17);
    constexpr uint32_t xrow = get_compile_time_arg_val(18);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(19);
    constexpr auto oargs = TensorAccessorArgs<20>();
    constexpr auto xargs = TensorAccessorArgs<oargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr uint32_t gw = 2 * npair;
    constexpr uint32_t pbytes = gw * 4096;
    constexpr uint32_t ncol = (cx1 - cx0 + 1) * (cy1 - cy0 + 1);

    const uint32_t nop = get_arg_val<uint32_t>(0);
    const uint32_t J = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2), ly = get_arg_val<uint32_t>(3);
    const uint32_t oaddr = get_arg_val<uint32_t>(4);
    const uint32_t nown = get_arg_val<uint32_t>(5);
    const uint32_t rx = get_arg_val<uint32_t>(6), ry = get_arg_val<uint32_t>(7);
    const uint32_t xaddr = get_arg_val<uint32_t>(8), iaddr = get_arg_val<uint32_t>(9);
    const uint32_t NB = get_arg_val<uint32_t>(10);
    const uint32_t op_base = 11 + nown;
    const uint32_t blk_base = op_base + nop;
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, R * 32 * 4);

    // the row index table (R * 32 uint32): this expert's token rows, padding -> the zero row
    noc_async_read_page(0, is, idx_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    volatile tt_l1_ptr uint32_t* grant_l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_L));
    volatile tt_l1_ptr uint32_t* grant_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_R));
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    const uint64_t left_arr = get_noc_addr(lx, ly, get_semaphore(SEM_ARR_R));
    const uint64_t right_arr = get_noc_addr(rx, ry, get_semaphore(SEM_ARR_L));

    // ---- Q1 state
    uint32_t g_e = 0, g_b = 0;  // next own block (expert, index)
    const bool q1_any = NB > 0;
    // ---- Q2 state
    uint32_t o_e = 0, o_p = 0, produced = 0, consumed = 0, e_start = 0, nl = 0, nr = 0;
    bool flushing = false;

    auto q1 = [&]() -> bool {  // true: did work
        if (!q1_any || g_e >= NEXP || !cb_pages_reservable_at_back(xrm_cb, BK)) {
            return false;
        }
        const uint32_t r = get_arg_val<uint32_t>(blk_base + 3 * g_b);
        const uint32_t k = get_arg_val<uint32_t>(blk_base + 3 * g_b + 1);
        const uint32_t n = get_arg_val<uint32_t>(blk_base + 3 * g_b + 2);
        cb_reserve_back(xrm_cb, BK);
        uint32_t l1 = get_write_ptr(xrm_cb);
#ifndef WX_NO_GATHER
        for (uint32_t i = 0; i < 32; ++i, l1 += BK * 64) {
            noc_async_read(xs.get_noc_addr(idx[r * 32 + i]) + k * 64, l1, n * 64);
        }
        noc_async_read_barrier();
#endif
        cb_push_back(xrm_cb, BK);
        if (++g_b == NB) {
            g_b = 0;
            ++g_e;
        }
        return true;
    };
    auto drain_y = [&](uint32_t q) {
        const uint32_t e = q / nown;
        if (e + 1 == NEXP) {
            const uint32_t row = get_arg_val<uint32_t>(11 + q % nown);
            uint32_t l1 = get_read_ptr(y_cb);
            for (uint32_t t = 0; t < DCOLS; ++t, l1 += obytes) {
                noc_async_write_page(row * HT + J * DCOLS + t, os, l1);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(y_cb, DCOLS);
    };
    auto q2 = [&]() -> bool {
        if (flushing) {
            if (consumed < produced && cb_pages_available_at_front(y_cb, DCOLS)) {
                drain_y(consumed++);
                return true;
            }
            return false;
        }
        const uint32_t fl = get_arg_val<uint32_t>(op_base + o_p);
        bool done = false;
        if (fl & F_TIL) {
            done = true;
        } else if (fl & F_DN) {
            if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                if (cb_pages_available_at_front(y_cb, DCOLS)) {
                    drain_y(consumed++);
                    done = true;
                }
            } else {
                done = true;
            }
        } else if (fl & (F_SEND_L | F_SEND_R)) {
            const bool left = fl & F_SEND_L;
            uint32_t& n = left ? nl : nr;
            invalidate_l1_cache();
            if (cb_pages_available_at_front(pout_cb, gw) && *(left ? grant_l : grant_r) >= n + 1) {
                const uint64_t ring = left ? get_noc_addr(lx, ly, pinR) : get_noc_addr(rx, ry, pinL);
                noc_async_write(get_read_ptr(pout_cb), ring + (n & 1) * pbytes, pbytes);
                noc_async_write_barrier();
                noc_semaphore_inc(left ? left_arr : right_arr, 1);
                cb_pop_front(pout_cb, gw);
                ++n;
                done = true;
            }
        } else if (cb_pages_available_at_front(h_cb, npair)) {
            const uint32_t slot = produced % NS;
            *scr = produced + 1;
            noc_async_write_multicast_loopback_src(
                get_read_ptr(h_cb),
                get_noc_multicast_addr(cx0, cy0, cx1, cy1, hring + (slot * KTD + J * npair) * htile),
                npair * htile,
                ncol,
                true /*linked*/);
            noc_semaphore_set_multicast_loopback_src(
                scr_addr, get_noc_multicast_addr(cx0, cy0, cx1, cy1, hflags + (slot * GC + J) * 16), ncol);
            noc_async_write_barrier();
            cb_pop_front(h_cb, npair);
            ++produced;
            done = true;
        }
        if (done && ++o_p == nop) {
            o_p = 0;
            if (++o_e == NEXP) {
                flushing = true;
            } else {
                e_start = produced;
            }
        }
        return done;
    };
    while (true) {
        const bool a = q1();
        const bool b = q2();
        if (!a && !b && (g_e >= NEXP || !q1_any) && flushing && consumed >= produced) {
            break;
        }
    }
    noc_async_atomic_barrier();
}
