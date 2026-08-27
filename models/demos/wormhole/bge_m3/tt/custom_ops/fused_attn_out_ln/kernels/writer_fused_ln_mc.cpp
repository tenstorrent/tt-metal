// SPDX-License-Identifier: Apache-2.0
// Multi-core writer: each core drains its M-slice of cb_out to the output tensor.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_out = tt::CBIndex::c_2;
void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t m_start_tile = get_arg_val<uint32_t>(a++);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t M_block = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t out_tb = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tb);
    for (uint32_t mb = 0; mb < per_core_M_blocks; mb++) {
        uint32_t m_tile = m_start_tile + mb * M_block;
        cb_wait_front(cb_out, M_block * N_tiles);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t m = 0; m < M_block; m++) {
            for (uint32_t n = 0; n < N_tiles; n++) {
                noc_async_write_page((m_tile + m) * N_tiles + n, out_acc, rp);
                rp += out_tb;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, M_block * N_tiles);
    }
}
