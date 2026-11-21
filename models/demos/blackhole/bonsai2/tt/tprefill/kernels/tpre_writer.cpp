// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Exact ternary prefill linear, writer (NCRISC): this core's output tiles (row tile m, columns n_first ..) to DRAM.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t Nt = get_common_arg_val<uint32_t>(1);
    const uint32_t Mt = get_common_arg_val<uint32_t>(2);
    const uint32_t n_first = get_arg_val<uint32_t>(0);
    const uint32_t n_count = get_arg_val<uint32_t>(1);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, bytes);
#ifdef GLU_PAIRS
    // one output tile per (gate, up) column pair: Nt is the output's tile width, n_first and n_count are even
    const uint32_t o_first = n_first / 2, o_count = n_count / 2;
#else
    const uint32_t o_first = n_first, o_count = n_count;
#endif
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < o_count; ++n) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(m * Nt + o_first + n), bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
