// SPDX-License-Identifier: Apache-2.0
// tp32pf floor microbench NCRISC1: writes MT*4 tiles per (expert, pass) to out[(e*PASSES+p)*MT*4 ..]
// CT: 0 cb_o, 1 MT, 2 PASSES, 3 NE, 4 NC, TensorAccessorArgs(out). RT: 0 core idx. Common: 0 out addr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t PASSES = get_compile_time_arg_val(2);
    constexpr uint32_t NE = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();
    const uint32_t me = get_arg_val<uint32_t>(0);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t e = me; e < NE; e += NC) {
        for (uint32_t p = 0; p < PASSES; ++p) {
            cb_wait_front(cb_o, MT * 4);
            uint32_t rd = get_read_ptr(cb_o);
            for (uint32_t i = 0; i < MT * 4; ++i) {
                noc_async_write(rd + i * 2048, oa.get_noc_addr((e * PASSES + p) * MT * 4 + i), 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_o, MT * 4);
        }
    }
}
