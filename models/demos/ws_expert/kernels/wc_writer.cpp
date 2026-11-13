// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert column-cluster sender (BRISC, NOC0), in the compute's order:
//   SEND_L / SEND_R piece: fp32 partial (CB 4) -> the left / right neighbour cluster's core in this row (its 2-slot
//               ring for partials from this side, after its grant), then its ARRIVED semaphore for this side += 1.
//   h piece:    the row's NPAIR h tiles (CB 16) -> slot (q % NS) of every cluster core's h ring (this core included),
//               as a linked multicast closed by the flag word (q + 1) for this core's pair group (mcast helper
//               pattern: no barrier between payload and flag); one write barrier before CB 16 / the flag source are
//               reused. Then the down output of row q - 1 (CB 17): the last expert's rows go to DRAM.
// No slot credits: a core sends h(q) after its own down(q - 2), which needed every peer's h(q - 2), which each peer
//   sent after its down(q - 4): with NS >= 4 slot q % NS is free everywhere.
// RT: 0 npiece, 1 J, 2 left x, 3 left y, 4 out addr, 5 nown, 6 right x, 7 right y, 8.. own[nown] (owned rows in
//     order), then per op: flags (1 PRE_L, 2 PRE_R, 4 SEND_L, 8 SEND_R, 16 DN)
// CT: 0 NPAIR, 1 NS, 2 KTD, 3 GC, 4 NEXP, 5 hring, 6 flags, 7 pinL (partials from the left), 8-11 column rect NOC0 (x0, y0, x1, y1), 12 h tile
//     bytes, 13 DCOLS, 14 HT (out tiles a row), 15 pinR (partials from the right), then TensorAccessorArgs(out)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3, SEM_SCR = 4;

void kernel_main() {
    constexpr uint32_t npair = get_compile_time_arg_val(0);
    constexpr uint32_t NS = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t GC = get_compile_time_arg_val(3);
    constexpr uint32_t NEXP = get_compile_time_arg_val(4);
    constexpr uint32_t hring = get_compile_time_arg_val(5);
    constexpr uint32_t flags = get_compile_time_arg_val(6);
    constexpr uint32_t pinL = get_compile_time_arg_val(7);
    constexpr uint32_t cx0 = get_compile_time_arg_val(8), cy0 = get_compile_time_arg_val(9);
    constexpr uint32_t cx1 = get_compile_time_arg_val(10), cy1 = get_compile_time_arg_val(11);
    constexpr uint32_t htile = get_compile_time_arg_val(12);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(13);
    constexpr uint32_t HT = get_compile_time_arg_val(14);
    constexpr uint32_t pinR = get_compile_time_arg_val(15);
    constexpr auto oargs = TensorAccessorArgs<16>();
    constexpr uint32_t gw = 2 * npair;
    constexpr uint32_t pbytes = gw * 4096;

    const uint32_t npiece = get_arg_val<uint32_t>(0);
    const uint32_t J = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2), ly = get_arg_val<uint32_t>(3);
    const uint32_t rx = get_arg_val<uint32_t>(6), ry = get_arg_val<uint32_t>(7);
    const uint32_t oaddr = get_arg_val<uint32_t>(4);
    const uint32_t nown = get_arg_val<uint32_t>(5);
    const uint32_t pbase = 8 + nown;
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);

    volatile tt_l1_ptr uint32_t* grant_l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_L));
    volatile tt_l1_ptr uint32_t* grant_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_R));
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    // (a partial sent left lands in the left core's from-the-right ring, and the other way round)
    const uint64_t left_arr = get_noc_addr(lx, ly, get_semaphore(SEM_ARR_R));
    const uint64_t right_arr = get_noc_addr(rx, ry, get_semaphore(SEM_ARR_L));
    constexpr uint32_t ncol = (cx1 - cx0 + 1) * (cy1 - cy0 + 1);

    auto drain_y = [&](uint32_t q) {  // the down output of the cluster's row q
        cb_wait_front(y_cb, DCOLS);
        const uint32_t e = q / nown;
        if (e + 1 == NEXP) {
            const uint32_t row = get_arg_val<uint32_t>(8 + q % nown);
            uint32_t l1 = get_read_ptr(y_cb);
            for (uint32_t t = 0; t < DCOLS; ++t) {
                noc_async_write_page(row * HT + J * DCOLS + t, os, l1);
                l1 += obytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(y_cb, DCOLS);
    };

    uint32_t nl = 0, nr = 0, q = 0, consumed = 0;
    auto send_partial = [&](volatile tt_l1_ptr uint32_t* grant, uint32_t& n, uint64_t dst_ring, uint64_t arr) {
        cb_wait_front(pout_cb, gw);
        noc_semaphore_wait_min(grant, n + 1);
        noc_async_write(get_read_ptr(pout_cb), dst_ring + (n & 1) * pbytes, pbytes);
        noc_async_write_barrier();
        noc_semaphore_inc(arr, 1);
        cb_pop_front(pout_cb, gw);
        ++n;
    };
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t e_start = q;  // (DN_LAG: DN ops take rows of earlier experts or >= DN_LAG productions old)
        for (uint32_t p = 0; p < npiece; ++p) {
            const uint32_t fl = get_arg_val<uint32_t>(pbase + p);
            if (fl & 16) {
                if (consumed < q && (!DN_LAG || consumed < e_start || consumed + DN_LAG < q)) {
                    drain_y(consumed++);
                }
                continue;
            }
            if (fl & 4) {
                send_partial(grant_l, nl, get_noc_addr(lx, ly, pinR), left_arr);
            } else if (fl & 8) {
                send_partial(grant_r, nr, get_noc_addr(rx, ry, pinL), right_arr);
            } else {
                const uint32_t slot = q % NS;
                cb_wait_front(h_cb, npair);
                *scr = q + 1;
#ifndef WC_NOSYNC
                noc_async_write_multicast_loopback_src(
                    get_read_ptr(h_cb),
                    get_noc_multicast_addr(cx0, cy0, cx1, cy1, hring + (slot * KTD + J * npair) * htile),
                    npair * htile,
                    ncol,
                    true /*linked*/);
                noc_semaphore_set_multicast_loopback_src(
                    scr_addr, get_noc_multicast_addr(cx0, cy0, cx1, cy1, flags + (slot * GC + J) * 16), ncol);
                noc_async_write_barrier();
#endif
                cb_pop_front(h_cb, npair);
                ++q;
            }
        }
    }
    while (consumed < q) {
        drain_y(consumed++);
    }
    noc_async_atomic_barrier();
}
