// SPDX-License-Identifier: Apache-2.0
//
// Exercise 04 — reader kernel.
//
// Same as lesson 03, but this core only owns tiles [start_tile, start_tile + n).

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

    // TODO: loop over this core's slice instead of [0, n_tiles).
    //       The body is exactly your lesson-03 reader.
    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: reserve, issue both reads, one barrier, push both.
    }
}
