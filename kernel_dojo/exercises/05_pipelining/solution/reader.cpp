// SPDX-License-Identifier: Apache-2.0
//
// Exercise 05 — reader kernel, reference solution.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);

    constexpr auto a_args = TensorAccessorArgs<3>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const uint32_t end_tile = start_tile + n_tiles;

    for (uint32_t i = start_tile; i < end_tile; i += block) {
        cb_reserve_back(cb_a, block);
        cb_reserve_back(cb_b, block);

        // One call per block: this is the start of the whole reserved run, and
        // it does not advance as pages are filled.
        const uint32_t base_a = get_write_ptr(cb_a);
        const uint32_t base_b = get_write_ptr(cb_b);

        // `block` is a compile-time constant, so this unrolls and the offsets
        // fold into immediates.
        for (uint32_t t = 0; t < block; t++) {
            noc_async_read_page(i + t, a, base_a + t * tile_bytes);
            noc_async_read_page(i + t, b, base_b + t * tile_bytes);
        }
        // 2*block transactions were in flight together; their latencies
        // overlapped instead of being paid one after another.
        noc_async_read_barrier();

        cb_push_back(cb_a, block);
        cb_push_back(cb_b, block);
    }
}
