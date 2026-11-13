// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert dynamic-M BRISC (NOC0). Plans every expert for this core from its M (ws_plan.hpp) into the 3-slot plan
// ring (a slot is reused once all readers - compute's three threads, the NCRISC and this RISC - finished the expert
// three back), loading the expert's row index table beside it. Then a non-blocking loop over:
//   P  planning (as far ahead as the ring allows),
//   Q1 x gather: this core's x blocks in stream order (plan's own blocks): 32 token rows x n K-tiles into CB 7,
//   Q2 compute outputs in each expert's program order: partials out, h multicasts, DN ops' down outputs (the last
//      expert's to DRAM).
// RT: 0 J, 1 left x, 2 left y, 3 right x, 4 right y, 5 out addr, 6 x addr, 7 idx addr, 8 counts addr, 9 k (cluster)
// CT: 0 NPAIR, 1 NS, 2 KTD, 3 GC, 4 NEXP, 5 hring, 6 hflags, 7 pinL, 8-11 column rect NOC0, 12 h tile bytes,
//     13 DCOLS, 14 HT, 15 pinR, 16 BK, 17 RMAX, 18 x row bytes, 19 idx L1 (3 slots of RMAX * 32 words),
//     20 plan buffer, 21 counts L1, 22 KT, 23 NCL, then TensorAccessorArgs(out), (x), (idx), (counts)
#ifndef DN_LAG
#define DN_LAG 0
#endif
#include "api/dataflow/dataflow_api.h"
#include "ws_plan.hpp"

constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t SEM_GRANT_L = 0, SEM_GRANT_R = 1, SEM_ARR_L = 2, SEM_ARR_R = 3, SEM_SCR = 4;
using namespace ws_plan;

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
    constexpr uint32_t RMAX = get_compile_time_arg_val(17);
    constexpr uint32_t xrow = get_compile_time_arg_val(18);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(19);
    constexpr uint32_t planbuf = get_compile_time_arg_val(20);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(21);
    constexpr uint32_t KT = get_compile_time_arg_val(22);
    constexpr uint32_t NCL = get_compile_time_arg_val(23);
    constexpr auto oargs = TensorAccessorArgs<24>();
    constexpr auto xargs = TensorAccessorArgs<oargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr uint32_t gw = 2 * npair;
    constexpr uint32_t pbytes = gw * 4096;
    constexpr uint32_t ncol = (cx1 - cx0 + 1) * (cy1 - cy0 + 1);

    const uint32_t J = get_arg_val<uint32_t>(0);
    const uint32_t lx = get_arg_val<uint32_t>(1), ly = get_arg_val<uint32_t>(2);
    const uint32_t rx = get_arg_val<uint32_t>(3), ry = get_arg_val<uint32_t>(4);
    const uint32_t oaddr = get_arg_val<uint32_t>(5), xaddr = get_arg_val<uint32_t>(6);
    const uint32_t iaddr = get_arg_val<uint32_t>(7), caddr = get_arg_val<uint32_t>(8);
    const uint32_t K = get_arg_val<uint32_t>(9);
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);
    const Geo geo{KT, gw, DCOLS * KTD, NCL, GC, BK};

    noc_async_read_page(0, cs, counts_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(planbuf);
    volatile tt_l1_ptr uint32_t* ctrl = plan + CTRL;
    auto slot = [&](uint32_t e) { return plan + (e % NSLOT) * SLOT_WORDS; };
    auto idx_of = [&](uint32_t e) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1 + (e % NSLOT) * RMAX * 32 * 4);
    };

    volatile tt_l1_ptr uint32_t* grant_l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_L));
    volatile tt_l1_ptr uint32_t* grant_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT_R));
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    const uint64_t left_arr = get_noc_addr(lx, ly, get_semaphore(SEM_ARR_R));
    const uint64_t right_arr = get_noc_addr(rx, ry, get_semaphore(SEM_ARR_L));

    // ---- P: planning
    uint32_t planned = 0;
    uint32_t g_e = 0, g_b = 0;                                    // Q1: next own block (expert, index)
    uint32_t o_e = 0, o_p = 0, produced = 0, consumed = 0, e_start = 0, nl = 0, nr = 0;  // Q2
    uint32_t out_e[8], out_i[8], e_prod = 0;  // produced rows not yet drained: (expert, index in the expert's own rows)
    auto my_done = [&]() { return (g_e < o_e ? g_e : o_e); };
    auto plan_step = [&]() -> bool {
        if (planned >= NEXP) {
            return false;
        }
        if (planned >= NSLOT) {
            const uint32_t need = planned - NSLOT + 1;  // readers finished expert planned - NSLOT
            invalidate_l1_cache();
            if (my_done() < need) {
                return false;
            }
            for (uint32_t d = DONE_NC; d < NDONE; ++d) {
                if (ctrl[d] < need) {
                    return false;
                }
            }
        }
        volatile tt_l1_ptr uint32_t* p = slot(planned);
        p[H_READY] = 0;
        make(geo, counts[planned], K, J, p);
        noc_async_read_page(planned, is, (uint32_t)idx_of(planned));
        noc_async_read_barrier();
        asm volatile("fence" ::: "memory");
        p[H_READY] = planned + 1;
        ++planned;
        return true;
    };

    auto q1 = [&]() -> bool {
        while (g_e < NEXP && g_e < planned && g_b >= slot(g_e)[H_NMY]) {  // (experts with no own blocks)
            ++g_e;
            g_b = 0;
        }
        if (g_e >= NEXP || g_e >= planned || !cb_pages_reservable_at_back(xrm_cb, BK)) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* p = slot(g_e);
        const uint32_t r = p[OFF_MY + 4 * g_b + 1], k = p[OFF_MY + 4 * g_b + 2], n = p[OFF_MY + 4 * g_b + 3];
        volatile tt_l1_ptr uint32_t* idx = idx_of(g_e);
        cb_reserve_back(xrm_cb, BK);
        uint32_t l1 = get_write_ptr(xrm_cb);
        for (uint32_t i = 0; i < 32; ++i, l1 += BK * 64) {
            noc_async_read(xs.get_noc_addr(idx[r * 32 + i]) + k * 64, l1, n * 64);
        }
        noc_async_read_barrier();
        cb_push_back(xrm_cb, BK);
        ++g_b;
        return true;
    };
    auto drain_y = [&]() {
        const uint32_t e = out_e[consumed % 8], i = out_i[consumed % 8];
        if (e + 1 == NEXP) {
            const uint32_t row = slot(e)[OFF_OWN + i];
            uint32_t l1 = get_read_ptr(y_cb);
            for (uint32_t t = 0; t < DCOLS; ++t, l1 += obytes) {
                noc_async_write_page(row * HT + J * DCOLS + t, os, l1);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(y_cb, DCOLS);
        ++consumed;
    };
    auto q2 = [&]() -> bool {
        if (o_e >= NEXP) {
            if (consumed < produced && cb_pages_available_at_front(y_cb, DCOLS)) {
                drain_y();
                return true;
            }
            return false;
        }
        if (o_e >= planned) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* p = slot(o_e);
        const uint32_t nop = p[H_NOP];
        bool done = false;
        if (o_p < nop) {
            const uint32_t fl = p[OFF_OP + 3 * o_p + 2];
            if (fl & DN) {
                if (consumed < produced && (!DN_LAG || consumed < e_start || consumed + DN_LAG < produced)) {
                    if (cb_pages_available_at_front(y_cb, DCOLS)) {
                        drain_y();
                        done = true;
                    }
                } else {
                    done = true;
                }
            } else if (fl & (SEND_L | SEND_R)) {
                const bool left = fl & SEND_L;
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
                const uint32_t hs = produced % NS;
                *scr = produced + 1;
                noc_async_write_multicast_loopback_src(
                    get_read_ptr(h_cb),
                    get_noc_multicast_addr(cx0, cy0, cx1, cy1, hring + (hs * KTD + J * npair) * htile),
                    npair * htile,
                    ncol,
                    true /*linked*/);
                noc_semaphore_set_multicast_loopback_src(
                    scr_addr, get_noc_multicast_addr(cx0, cy0, cx1, cy1, hflags + (hs * GC + J) * 16), ncol);
                noc_async_write_barrier();
                cb_pop_front(h_cb, npair);
                out_e[produced % 8] = o_e;
                out_i[produced % 8] = e_prod++;
                ++produced;
                done = true;
            }
            if (done) {
                ++o_p;
            }
        }
        if (o_p >= nop) {  // expert done
            o_p = 0;
            e_prod = 0;
            ++o_e;
            e_start = produced;
            done = true;
        }
        return done;
    };
    while (true) {
        const bool a = plan_step();
        const bool b = q1();
        const bool c = q2();
        ctrl[DONE_BR] = my_done();
        if (!a && !b && !c && planned >= NEXP && g_e >= NEXP && o_e >= NEXP && consumed >= produced) {
            break;
        }
    }
    noc_async_atomic_barrier();
}
