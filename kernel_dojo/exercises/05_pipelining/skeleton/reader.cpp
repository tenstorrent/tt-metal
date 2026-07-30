// SPDX-License-Identifier: Apache-2.0
//
// Exercise 05 — reader kernel, blocked.
//
// Read a whole block of tiles per iteration, with all reads of the block in
// flight at once, so their DRAM latencies overlap.

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

    // Bytes per page — needed to step through the reserved run of pages.
    const uint32_t tile_bytes = get_tile_size(cb_a);

    const uint32_t end_tile = start_tile + n_tiles;

    // n_tiles is guaranteed to be a multiple of `block`.
    for (uint32_t i = start_tile; i < end_tile; i += block) {
        // TODO: reserve `block` pages in each of cb_a and cb_b.

        // TODO: take the base write pointer of each CB *once*, before the
        //       inner loop — it points at the start of the reserved run.

        // TODO: inner loop over t in [0, block): issue the read of tile i+t of
        //       `a` to base_a + t * tile_bytes, and likewise for `b`.

        // TODO: a single barrier for all 2*block transactions.

        // TODO: push `block` pages to each CB.
    }
}
