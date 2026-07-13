// SPDX-License-Identifier: Apache-2.0
// Regime-A prototype writer: drain the single output block out[M_block, N_block] to interleaved DRAM.
// compute.cpp pushes out one M-row (N_block tiles) at a time. out tile(m,n) id = m*Nt + n.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t N_block = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t m0 = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t out_cb = 2;

    for (uint32_t m = 0; m < M_block; ++m) {
        cb_wait_front(out_cb, N_block);
        uint32_t r = get_read_ptr(out_cb);
        for (uint32_t n = 0; n < N_block; ++n) {
            noc_async_write_page((m0 + m) * Nt + (n0 + n), out, r);
            r += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, N_block);
    }
}
