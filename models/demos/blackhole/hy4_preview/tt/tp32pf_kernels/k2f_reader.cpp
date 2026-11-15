// SPDX-License-Identifier: Apache-2.0
// tp32pf K2f (fused passes; LAYOUT v1 W2) NCRISC data in, ALL 6 passes S in one launch (loop S outside the unit loop;
// weights of the next (S, unit) are prefetched across pass boundaries); per pass S (= column group s of the v1 order;
// output columns 24b + 4S + n, b < 8, n < 4): plan page; per unit the expert's down slice of pass S (8 banks x 8 tiles
// [k2][n4] = 64 tiles, tile index ((S*256+e)*8) per bank) resident in cb_w (double buffered, one unit ahead, trid 1);
// act tiles (mt,k) of the unit's M-tile pairs (4 tiles per pair) to cb_a. CT: 0 cb_w, 1 cb_a, 2 cb_meta, 3 S,
// TensorAccessorArgs(plan), (act). RT: 0 me. Common: 0 w2 addr, 1 plan addr, 2 act addr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t NPASS = get_compile_time_arg_val(3);
    constexpr auto p_args = TensorAccessorArgs<4>();
    constexpr auto a_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 576, RUN = 8 * TB;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t w2 = get_common_arg_val<uint32_t>(0);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto aa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(2), 2048);
    cb_reserve_back(cb_meta, 1);
    const uint32_t ml = get_write_ptr(cb_meta);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    auto issue_w = [&](uint32_t S, uint32_t e) {
        cb_reserve_back(cb_w, 64);
        const uint32_t wr = get_write_ptr(cb_w);
        noc_async_read_set_trid(1);
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t b = (j + me) & 7;
            const uint64_t base = get_noc_addr_from_bank_id<true>(b, w2);
            noc_async_read_one_packet_set_state<true>(base, RUN, (j + me) & 3);
            noc_async_read_one_packet_with_state_with_trid(base, (S * 256 + e) * RUN, wr + b * RUN, 1);
        }
    };
    const uint32_t total = NPASS * nu;
    if (total > 0) {
        issue_w(0, plan[1]);
    }
    for (uint32_t it = 0; it < total; ++it) {
        const uint32_t S = it / nu, u = it - S * nu;
        noc_async_read_barrier_with_trid(1);
        cb_push_back(cb_w, 64);
        if (it + 1 < total) {
            const uint32_t S2 = (it + 1) / nu, u2 = (it + 1) - S2 * nu;
            issue_w(S2, plan[1 + 3 * u2]);
        }
        const uint32_t mt0 = plan[2 + 3 * u], mtc = plan[3 + 3 * u];
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            cb_reserve_back(cb_a, 4);
            const uint32_t al = get_write_ptr(cb_a);
            noc_async_read_set_trid(3);
            for (uint32_t i = 0; i < 4; ++i) {
                const uint64_t na = aa.get_noc_addr((mt0 + 2 * pr) * 2 + i);
                noc_async_read_one_packet_set_state<true>(na, 2048, 2);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)na, 0, al + i * 2048, 3);
            }
            noc_async_read_barrier_with_trid(3);
            cb_push_back(cb_a, 4);
        }
    }
    noc_async_read_barrier();
}
