// SPDX-License-Identifier: Apache-2.0
//
// Exercise 05 — writer kernel (provided).
//
// Already blocked — read it as a worked example of the same batching idea on
// the write side, then apply the pattern to your reader.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t block = get_compile_time_arg_val(1);

    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t tile_bytes = get_tile_size(cb_in);
    const uint32_t end_tile = start_tile + n_tiles;

    for (uint32_t i = start_tile; i < end_tile; i += block) {
        // Wait for the whole block, so all its writes can be issued together.
        cb_wait_front(cb_in, block);
        const uint32_t base = get_read_ptr(cb_in);

        for (uint32_t t = 0; t < block; t++) {
            noc_async_write_page(i + t, dst, base + t * tile_bytes);
        }
        // One barrier drains all `block` writes.
        noc_async_write_barrier();

        cb_pop_front(cb_in, block);
    }
}
