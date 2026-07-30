// SPDX-License-Identifier: Apache-2.0
//
// Exercise 04 — reader kernel, reference solution.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    // Only this core's slice. Every core runs this same binary; the runtime
    // args are what make them do different work.
    const uint32_t end_tile = start_tile + n_tiles;

    for (uint32_t i = start_tile; i < end_tile; i++) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        noc_async_read_page(i, a, get_write_ptr(cb_a));
        noc_async_read_page(i, b, get_write_ptr(cb_b));
        noc_async_read_barrier();

        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
