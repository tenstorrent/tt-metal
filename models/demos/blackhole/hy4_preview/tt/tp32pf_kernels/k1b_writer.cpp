// SPDX-License-Identifier: Apache-2.0
// tp32pf K1b NCRISC1: scaled act tiles [mt][c] (2 per M-tile) to act[(mt_global)*2 + c]. Reads its own copy of the plan
// page. CT: 0 cb_o, 1 cb_ws (scratch 1 page of 2048 B), TensorAccessorArgs(plan), TensorAccessorArgs(out). RT: 0 me.
// Common: 0 plan addr, 1 out addr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ws = get_compile_time_arg_val(1);
    constexpr auto p_args = TensorAccessorArgs<2>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), 2048);
    cb_reserve_back(cb_ws, 1);
    const uint32_t ml = get_write_ptr(cb_ws);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mt0 = plan[2 + 3 * u], mtc = plan[3 + 3 * u];
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
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
