// SPDX-License-Identifier: Apache-2.0
// Cross-core LN reduce PROBE writer. Drains cb_out (this core's N-slice, obn tiles)
// to the output DRAM tensor at tiles (m, n_start+ns).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = tt::CBIndex::c_2;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n_start = get_arg_val<uint32_t>(a++);

    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ns = get_compile_time_arg_val(1);
    constexpr uint32_t M_t = get_compile_time_arg_val(2);
    constexpr uint32_t tb = get_compile_time_arg_val(3);
    constexpr uint32_t obn = M_t * Ns;

    constexpr auto o_args = TensorAccessorArgs<4>();
    const auto o_acc = TensorAccessor(o_args, out_addr, tb);

    cb_wait_front(cb_out, obn);
    uint32_t rp = get_read_ptr(cb_out);
    for (uint32_t m = 0; m < M_t; m++) {
        for (uint32_t n = 0; n < Ns; n++) {
            noc_async_write_page(m * N_tiles + n_start + n, o_acc, rp);
            rp += tb;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, obn);
}
