// SPDX-License-Identifier: Apache-2.0
//
// Exercise 03 — reader kernel.
//
// Read tile i of tensor `a` into cb_a and tile i of tensor `b` into cb_b, for
// each of n_tiles tiles. Try to keep both reads in flight at once.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    // The accessor args for `a` start at compile-time arg 2. Those for `b`
    // start wherever a's args ended — ask, don't hard-code.
    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);

    // TODO: build the accessor for `b`, starting at
    //       a_args.next_compile_time_args_offset().

    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: reserve one page in each of cb_a and cb_b.

        // TODO: issue both reads (tile i of a, tile i of b) back to back.

        // TODO: one barrier drains both.

        // TODO: push one page into each CB.
    }
}
