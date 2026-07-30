// SPDX-License-Identifier: Apache-2.0
//
// Exercise 03 — reader kernel, reference solution.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);
    // Start b's args where a's ended; the count varies with the memory config.
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        // Both reads are issued before either is waited on, so the two DRAM
        // round trips overlap instead of serialising.
        noc_async_read_page(i, a, get_write_ptr(cb_a));
        noc_async_read_page(i, b, get_write_ptr(cb_b));
        noc_async_read_barrier();  // drains every outstanding read, not just one

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
