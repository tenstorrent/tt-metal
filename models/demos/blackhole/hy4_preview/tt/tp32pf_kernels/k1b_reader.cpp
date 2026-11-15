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
    constexpr auto p_args = TensorAccessorArgs<5>();
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

    auto issue_w = [&](uint32_t e) {
        cb_reserve_back(cb_w, 768);
        const uint32_t wr = get_write_ptr(cb_w);
        noc_async_read_set_trid(1);
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t b = (j + me) & 7;
            const uint64_t base = get_noc_addr_from_bank_id<true>(b, w01 + e * BLK);
            noc_async_read_one_packet_set_state<true>(base, PA, (j + me) & 3);
            for (uint32_t i = 0; i < 4; ++i) {
                noc_async_read_one_packet_with_state_with_trid(base, i * PA, wr + b * BLK + i * PA, 1);
            }
        }
    };
    if (nu > 0) {
        issue_w(plan[1]);
    }
    for (uint32_t u = 0; u < nu; ++u) {
        noc_async_read_barrier_with_trid(1);
        cb_push_back(cb_w, 768);
        if (u + 1 < nu) {
            issue_w(plan[1 + 3 * (u + 1)]);
        }
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
            if (me == 0 && u == 0 && pr == 0) {
                DPRINT(
                    "K1b core0 u0 e={} mt0={} mtc={} ix0={} ix1={} ix63={}\n",
                    (uint32_t)plan[1],
                    mt0,
                    mtc,
                    (uint32_t)ix[0],
                    (uint32_t)ix[1],
                    (uint32_t)ix[63]);
            }
            cb_push_back(cb_sc, 2);
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (j + me) & 7;
                for (uint32_t h = 0; h < 2; ++h) {
                    cb_reserve_back(cb_rm, 24);
                    const uint32_t xw = get_write_ptr(cb_rm);
                    noc_async_read_set_trid(2);
                    for (uint32_t i = 0; i < 32; ++i) {
                        const uint64_t na = xa.get_noc_addr(ix[32 * h + i], b * SEG);
                        noc_async_read_one_packet_set_state<true>(na, SEG, i & 3);
                        noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, xw + i * SEG, 2);
                    }
                    noc_async_read_barrier_with_trid(2);
                    cb_push_back(cb_rm, 24);
                }
            }
        }
    }
    noc_async_read_barrier();
}
