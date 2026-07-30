// SPDX-License-Identifier: Apache-2.0
//
// Exercise 06 — matmul writer (provided).
//
// Output tiles are produced in linear order (row-major over the Mt x Nt grid),
// so this is just the lesson-01 writer with n_tiles = Mt * Nt.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in, 1);
        noc_async_write_page(i, dst, get_read_ptr(cb_in));
        noc_async_write_barrier();
        cb_pop_front(cb_in, 1);
    }
}
