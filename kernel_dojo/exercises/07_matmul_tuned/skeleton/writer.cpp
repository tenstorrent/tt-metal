// SPDX-License-Identifier: Apache-2.0
//
// Exercise 07 — matmul writer (provided).
//
// This core produces output tiles for rows [start_row, start_row + n_rows), so
// its slice of C is the contiguous run starting at start_row * Nt.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t end_tile = start_tile + n_tiles;

    for (uint32_t i = start_tile; i < end_tile; i++) {
        cb_wait_front(cb_in, 1);
        noc_async_write_page(i, dst, get_read_ptr(cb_in));
        noc_async_write_barrier();
        cb_pop_front(cb_in, 1);
    }
}
