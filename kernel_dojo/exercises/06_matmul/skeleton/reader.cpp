// SPDX-License-Identifier: Apache-2.0
//
// Exercise 06 — matmul reader.
//
// For each output tile (mt, nt), stream the Kt tiles of A's row mt into cb_a
// and the Kt tiles of B's column nt into cb_b, in matching order.
//
// Tile (r, c) of a matrix that is `cols` tiles wide lives at page r * cols + c.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);
    const uint32_t Kt = get_arg_val<uint32_t>(3);
    const uint32_t Nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: inner loop over kt in [0, Kt):
            //   - reserve a page in each CB
            //   - read A tile (mt, kt) and B tile (kt, nt)
            //   - barrier, then push both
        }
    }
}
