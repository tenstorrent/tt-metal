// SPDX-License-Identifier: Apache-2.0
// tp32pf K1b NCRISC (data in): plan page; per unit the expert's whole gate/up slice (768 bfp4 tiles = 8 bank runs)
// resident in cb_w (double buffered, one unit ahead, trid 1); per pair of M-tiles: gather index page (64 tokens, trid
// 3), 2 score tiles (trid 3), then for each K block b (24 tiles = 768 cols = 1536 B per row): 64 row segments of 1536 B
// from the row-major x_all (DRAM, page = one token row of 12288 B) into cb_rm (48 pages = 32 rows x 1536 B per tile
// row, tilized by compute), trid 2. CT: 0 cb_w, 1 cb_rm, 2 cb_meta, 3 cb_ix, 4 cb_sc, TensorAccessorArgs(plan), (gidx),
// (x), (sc). RT: 0 me. Common: 0 w01, 1 x, 2 plan, 3 gidx, 4 sc
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ix = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(4);
    constexpr uint32_t PERF = get_compile_time_arg_val(5);
    constexpr uint32_t XL1 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_xs = get_compile_time_arg_val(7);
    constexpr uint32_t sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t NC = get_compile_time_arg_val(9);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(11);
    constexpr uint32_t TPC = get_compile_time_arg_val(10);
    constexpr auto p_args = TensorAccessorArgs<12>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 576, BLK = 96 * TB, PA = 24 * TB, SEG = 1536, ROW = 12288;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t w01 = get_common_arg_val<uint32_t>(0);
    const uint32_t xaddr = get_common_arg_val<uint32_t>(1);
    const uint32_t paddr = get_common_arg_val<uint32_t>(2);
    const uint32_t gaddr = get_common_arg_val<uint32_t>(3);
    const uint32_t saddr = get_common_arg_val<uint32_t>(4);
    const uint32_t ttok = get_common_arg_val<uint32_t>(5);

    const auto pa = TensorAccessor(p_args, paddr, 2048);
    const auto ga = TensorAccessor(g_args, gaddr, 256);
    const auto xa = TensorAccessor(x_args, xaddr, ROW);
    const auto sa = TensorAccessor(s_args, saddr, 2048);
    cb_reserve_back(cb_meta, 1);
    const uint32_t ml = get_write_ptr(cb_meta);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_ix, 1);
    const uint32_t ixl = get_write_ptr(cb_ix);
    volatile tt_l1_ptr uint32_t* ix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ixl);

    uint32_t xs = 0;
    uint32_t tres = 0, tiss = 0, tbar = 0, npairs = 0;
    const uint32_t t_start = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
    if (XL1) {
        // stage this core's TPC token rows of x_all in L1, then wait until every core did (all-to-all semaphore)
        cb_reserve_back(cb_xs, TPC * 6);
        xs = get_write_ptr(cb_xs);
        for (uint32_t i = 0; i < TPC; ++i) {
            const uint32_t t = me * TPC + i;
            if (t < ttok) {
                noc_async_read(xa.get_noc_addr(t), xs + i * ROW, ROW);
            }
        }
        noc_async_read_barrier();
        const uint32_t sem_l1 = get_semaphore(sem_id);
        for (uint32_t q = 0; q < NC; ++q) {
            noc_semaphore_inc(
                get_noc_addr(get_common_arg_val<uint32_t>(6 + 2 * q), get_common_arg_val<uint32_t>(7 + 2 * q), sem_l1),
                1);
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1), NC);
    }

    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mt0 = plan[2 + 3 * u], mtc = plan[3 + 3 * u];
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            const uint32_t mt = mt0 + 2 * pr;
            // gather indices (64 x u32) and the 2 score tiles
            noc_async_read_set_trid(3);
            {
                const uint64_t na = ga.get_noc_addr(mt >> 1);
                noc_async_read_one_packet_set_state<true>(na, 256, 1);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, ixl, 3);
            }
            cb_reserve_back(cb_sc, 2);
            const uint32_t scl = get_write_ptr(cb_sc);
            for (uint32_t m = 0; m < 2; ++m) {
                const uint64_t na = sa.get_noc_addr(mt + m);
                noc_async_read_one_packet_set_state<true>(na, 2048, 2);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, scl + m * 2048, 3);
            }
            noc_async_read_barrier_with_trid(3);
            cb_push_back(cb_sc, 2);
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (j + me) & 7;
                for (uint32_t h = 0; h < (SPLIT ? 1 : 2); ++h) {
                    const uint32_t tA = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
                    cb_reserve_back(cb_rm, 24);
                    const uint32_t tB = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
                    const uint32_t xw = get_write_ptr(cb_rm);
                    noc_async_read_set_trid(2);
                    if (PERF & 16) {
                        const uint32_t o = (me + 1) % NC;
                        for (uint32_t q = 0; q < 3; ++q) {
                            const uint64_t na = get_noc_addr(
                                get_common_arg_val<uint32_t>(6 + 2 * o),
                                get_common_arg_val<uint32_t>(7 + 2 * o),
                                xs + q * 16384);
                            noc_async_read_one_packet_set_state<true>(na, 16384, q & 3);
                            noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, xw + q * 16384, 2);
                        }
                    }
                    if (PERF & 32) {
                        // timing probe: 8 reads of 6144 B (same bytes as 32 x 1536 B) from random owners
                        for (uint32_t i = 0; i < 8; ++i) {
                            const uint32_t g = ix[4 * i];
                            const uint32_t o = g >> 16, tl = g & 0xFFFF;
                            const uint64_t na = XL1 ? get_noc_addr(
                                                          get_common_arg_val<uint32_t>(6 + 2 * o),
                                                          get_common_arg_val<uint32_t>(7 + 2 * o),
                                                          xs + tl * ROW)
                                                    : xa.get_noc_addr(g);
                            noc_async_read_one_packet_set_state<true>(na, 6144, i & 3);
                            noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, xw + i * 6144, 2);
                        }
                    }
                    for (uint32_t i = 0; i < ((PERF & 49) ? 0 : 32); ++i) {
                        const uint32_t g = ix[32 * h + i];
                        uint64_t na;
                        if (XL1) {
                            uint32_t o = g >> 16, tl = g & 0xFFFF;
                            if (PERF & 4) {
                                o = me;
                                tl = i % TPC;
                            }
                            if (PERF & 8) {
                                o = (me + 1) % NC;
                                tl = i % TPC;
                            }
                            na = get_noc_addr(
                                get_common_arg_val<uint32_t>(6 + 2 * o),
                                get_common_arg_val<uint32_t>(7 + 2 * o),
                                xs + tl * ROW + b * SEG);
                        } else {
                            na = xa.get_noc_addr(g, b * SEG);
                        }
                        noc_async_read_one_packet_set_state<true>(na, SEG, i & 3);
                        noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, xw + i * SEG, 2);
                    }
                    const uint32_t tC = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
                    noc_async_read_barrier_with_trid(2);
                    const uint32_t tD = *reinterpret_cast<volatile uint32_t*>(0xFFB121F0);
                    tres += tB - tA;
                    tiss += tC - tB;
                    tbar += tD - tC;
                    cb_push_back(cb_rm, 24);
                }
            }
        }
    }
    noc_async_read_barrier();
    DPRINT(
        "K1b core {} nu={} total={} reserve={} issue={} barrier={}\n",
        me,
        nu,
        *reinterpret_cast<volatile uint32_t*>(0xFFB121F0) - t_start,
        tres,
        tiss,
        tbar);
}
