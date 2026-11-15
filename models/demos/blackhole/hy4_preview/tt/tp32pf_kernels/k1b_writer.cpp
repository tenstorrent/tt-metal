// SPDX-License-Identifier: Apache-2.0
// tp32pf K1b NCRISC1 (NoC 1): (SPLIT) the second 32-row half of every K block of the row gather (the first half is
// gathered by the reader on NoC 0, so both NoCs carry the response traffic) into cb_rm1, trid 2; then the scaled act
// tiles [mt][c] (2 per M-tile) to act[(mt_global) * 2 + c]. CT: 0 cb_o, 1 cb_ws (2 pages of 2048 B), 2 cb_rm1, 3 cb_xs,
// 4 XL1, 5 NC, 6 TPC, 7 SPLIT, 8 sem_id, TensorAccessorArgs(plan), (out), (gidx), (x). RT: 0 me. Common: 0 plan, 1 out,
// 2 gidx, 3 x, 4 token rows, 5.. NC x (x, y) physical cores
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ws = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xs = get_compile_time_arg_val(3);
    constexpr uint32_t XL1 = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t TPC = get_compile_time_arg_val(6);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(7);
    constexpr uint32_t sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_w = get_compile_time_arg_val(9);
    constexpr auto p_args = TensorAccessorArgs<10>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t SEG = 1536, ROW = 12288, TB = 576, BLK = 96 * TB, PA = 24 * TB;
    const uint32_t w01 = get_common_arg_val<uint32_t>(5);
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ga = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), 256);
    const auto xa = TensorAccessor(x_args, get_common_arg_val<uint32_t>(3), ROW);
    cb_reserve_back(cb_ws, 2);
    const uint32_t ml = get_write_ptr(cb_ws);
    const uint32_t ixl = ml + 2048;
    volatile tt_l1_ptr uint32_t* ix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ixl);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    uint32_t xs = 0;
    if (XL1 && SPLIT) {
        xs = get_write_ptr(cb_xs);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), NC);
    }
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
            if (SPLIT) {
                noc_async_read_set_trid(3);
                {
                    const uint64_t na = ga.get_noc_addr((mt0 + 2 * pr) >> 1);
                    noc_async_read_one_packet_set_state<true>(na, 256, 1);
                    noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, ixl, 3);
                }
                noc_async_read_barrier_with_trid(3);
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t b = (j + me) & 7;
                    cb_reserve_back(cb_rm1, 24);
                    const uint32_t xw = get_write_ptr(cb_rm1);
                    noc_async_read_set_trid(2);
                    for (uint32_t i = 0; i < 32; ++i) {
                        const uint32_t g = ix[32 + i];
                        uint64_t na;
                        if (XL1) {
                            const uint32_t o = g >> 16, tl = g & 0xFFFF;
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
                    noc_async_read_barrier_with_trid(2);
                    cb_push_back(cb_rm1, 24);
                }
            }
            cb_wait_front(cb_o, 4);
            const uint32_t rd = get_read_ptr(cb_o);
            for (uint32_t i = 0; i < 4; ++i) {
                noc_async_write(rd + i * 2048, oa.get_noc_addr((mt0 + 2 * pr) * 2 + i), 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_o, 4);
        }
    }
}
