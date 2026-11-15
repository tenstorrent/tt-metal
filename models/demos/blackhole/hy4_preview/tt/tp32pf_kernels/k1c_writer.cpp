// SPDX-License-Identifier: Apache-2.0
// tp32pf K1c NCRISC1 (NoC 1): the expert's whole gate/up slice (768 bfp4 tiles = 8 bank runs, trid 1) into cb_w (single
// buffered, next unit issued after the previous unit's act tiles are written = after compute is done with the buffer),
// and the scaled act tiles [mt][c] (2 per real M-tile) to act[mt * 2 + c]. CT: 0 cb_o2, 1 cb_ws, 2 cb_w,
// TensorAccessorArgs(plan), (act). RT: 0 me. Common: 0 plan, 1 act, 2 w01
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ws = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr auto p_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 576, BLK = 96 * TB, PA = 24 * TB;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), 2048);
    const uint32_t w01 = get_common_arg_val<uint32_t>(2);
    cb_reserve_back(cb_ws, 1);
    const uint32_t ml = get_write_ptr(cb_ws);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
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
    for (uint32_t u = 0; u < nu; ++u) {
        issue_w(plan[1 + 3 * u]);
        noc_async_read_barrier_with_trid(1);
        cb_push_back(cb_w, 768);
        const uint32_t mt0 = plan[2 + 3 * u], nreal = plan[304 + u];
        for (uint32_t m = 0; m < nreal; ++m) {
            cb_wait_front(cb_o, 2);
            const uint32_t rd = get_read_ptr(cb_o);
            for (uint32_t i = 0; i < 2; ++i) {
                noc_async_write(rd + i * 2048, oa.get_noc_addr((mt0 + m) * 2 + i), 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_o, 2);
        }
    }
}
