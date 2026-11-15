// SPDX-License-Identifier: Apache-2.0
// tp32pf K1c NCRISC (rows in, NoC 0): per M-tile of every unit (only the real, non-padding M-tiles: plan[304 + u]): the
// score tile and the gather index page, then 32 FULL token rows (12288 B each, one NoC read per row from the row-major
// x_all in DRAM, trid 2) into cb_rm (192 pages = 32 rows x 6 pages; tilized by compute). CT: 0 cb_rm, 1 cb_meta, 2
// cb_ix, 3 cb_sc, TensorAccessorArgs(plan), (gidx), (x), (sc). RT: 0 me. Common: 0 x, 1 plan, 2 gidx, 3 sc
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ix = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr auto p_args = TensorAccessorArgs<4>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t ROW = 12288;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto xa = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), ROW);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ga = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto sa = TensorAccessor(s_args, get_common_arg_val<uint32_t>(3), 2048);
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
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mt0 = plan[2 + 3 * u], nreal = plan[304 + u];
        for (uint32_t m = 0; m < nreal; ++m) {
            const uint32_t mt = mt0 + m;
            noc_async_read_set_trid(3);
            {
                // the 32 token ids of this M-tile: 128 B at word (mt * 32) of the row-major gidx table (pages of 512
                // words)
                const uint64_t na = ga.get_noc_addr(mt >> 4, (mt & 15) * 128);
                noc_async_read_one_packet_set_state<true>(na, 128, 1);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, ixl, 3);
            }
            {
                // the 32 bf16 scores of this M-tile: 64 B at u16 (mt * 32) of the score table (pages of 1024 u16)
                const uint64_t na = sa.get_noc_addr(mt >> 5, (mt & 31) * 64);
                noc_async_read_one_packet_set_state<true>(na, 64, 2);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, ixl + 256, 3);
            }
            noc_async_read_barrier_with_trid(3);
            cb_reserve_back(cb_sc, 1);
            {
                // column-0 broadcast tile: element (r, 0) of a 32x32 bf16 tile = u16 r * 16 (r < 16) / 512 + (r - 16) *
                // 16
                volatile tt_l1_ptr uint16_t* sv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ixl + 256);
                volatile tt_l1_ptr uint16_t* tv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_sc));
                for (uint32_t r = 0; r < 16; ++r) {
                    tv[r * 16] = sv[r];
                    tv[512 + r * 16] = sv[16 + r];
                }
            }
            cb_push_back(cb_sc, 1);
            const uint32_t half = 0;
            cb_reserve_back(cb_rm, 192);
            const uint32_t xw = get_write_ptr(cb_rm);
            noc_async_read_set_trid(2);
            for (uint32_t i = 0; i < 32; ++i) {
                const uint64_t na = xa.get_noc_addr(ix[half + i]);
                noc_async_read_one_packet_set_state<true>(na, ROW, i & 3);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, xw + i * ROW, 2);
            }
            noc_async_read_barrier_with_trid(2);
            cb_push_back(cb_rm, 192);
        }
    }
    noc_async_read_barrier();
}
