// SPDX-License-Identifier: Apache-2.0
//
// Exercise 06 — matmul reader, reference solution.

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
            // Stream A's row mt against B's column nt. Note this re-reads the
            // whole of B once per row of A — the inefficiency lesson 07 fixes.
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_reserve_back(cb_a, 1);
                cb_reserve_back(cb_b, 1);

                noc_async_read_page(mt * Kt + kt, a, get_write_ptr(cb_a));
                noc_async_read_page(kt * Nt + nt, b, get_write_ptr(cb_b));
                noc_async_read_barrier();

                cb_push_back(cb_a, 1);
                cb_push_back(cb_b, 1);
            }
        }
    }
}
