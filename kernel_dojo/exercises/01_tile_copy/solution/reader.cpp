// SPDX-License-Identifier: Apache-2.0
//
// Exercise 01 — reader kernel, reference solution.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // Block until the writer has drained a page for us to fill.
        cb_reserve_back(cb_out, 1);
        const uint32_t l1_addr = get_write_ptr(cb_out);

        noc_async_read_page(i, src, l1_addr);
        // The read is in flight, not complete. Nothing may observe the page
        // until this barrier returns.
        noc_async_read_barrier();

        cb_push_back(cb_out, 1);
    }
}
