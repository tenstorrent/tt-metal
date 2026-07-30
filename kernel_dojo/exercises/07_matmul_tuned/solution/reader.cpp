// SPDX-License-Identifier: Apache-2.0
//
// Exercise 07 — matmul reader with A-row reuse, reference solution.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);
    const uint32_t n_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);

    constexpr auto a_args = TensorAccessorArgs<2>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t end_row = start_row + n_rows;

    for (uint32_t mt = start_row; mt < end_row; mt++) {
        // A's row, read once and then reused across all Nt output tiles.
        cb_reserve_back(cb_a, Kt);
        const uint32_t base_a = get_write_ptr(cb_a);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_page(mt * Kt + kt, a, base_a + kt * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, Kt);

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // B's column nt: tiles (0..Kt-1, nt), stride Nt through the pages.
            cb_reserve_back(cb_b, Kt);
            const uint32_t base_b = get_write_ptr(cb_b);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                noc_async_read_page(kt * Nt + nt, b, base_b + kt * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_b, Kt);
        }
    }
}
