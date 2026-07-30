// SPDX-License-Identifier: Apache-2.0
//
// Exercise 01 — reader kernel (runs on NCRISC).
//
// Read `n_tiles` tiles from DRAM into circular buffer `cb_out`, one at a time.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    // Runtime args: where the source tensor lives, and how much of it is ours.
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // Compile-time arg 0 is the CB we produce into.
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    // Compile-time args 1.. describe how the source tensor is laid out in DRAM.
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: reserve one page of space in cb_out.

        // TODO: find the address of that page with get_write_ptr().

        // TODO: issue a read of tile `i` into it with noc_async_read_page().

        // TODO: wait for the read to land with noc_async_read_barrier().

        // TODO: publish the page with cb_push_back() so the writer can see it.
    }
}
