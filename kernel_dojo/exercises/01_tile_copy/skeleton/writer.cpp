// SPDX-License-Identifier: Apache-2.0
//
// Exercise 01 — writer kernel (runs on BRISC).
//
// Take `n_tiles` tiles out of circular buffer `cb_in` and write them to DRAM.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // Compile-time arg 0 is the CB we consume from.
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: wait until one page is available in cb_in.

        // TODO: find its address with get_read_ptr().

        // TODO: issue a write of that page to tile `i` with noc_async_write_page().

        // TODO: wait for the write to complete with noc_async_write_barrier().
        //       Skipping this frees the page while the NoC is still reading it.

        // TODO: release the page with cb_pop_front().
    }
}
