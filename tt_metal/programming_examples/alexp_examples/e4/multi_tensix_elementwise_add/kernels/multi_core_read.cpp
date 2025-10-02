// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    uint32_t core_tiles_start = get_arg_val<uint32_t>(0);  // Starting tile index for this core
    uint32_t core_num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles for this core
    uint32_t r_tiles = get_arg_val<uint32_t>(2);           // Number of B tiles (replicated)

    constexpr uint32_t cb_in0 = tt::CB::c_in0;  // A tiles (distributed)
    constexpr uint32_t cb_in1 = tt::CB::c_in1;  // B tiles (replicated)

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create tensor accessors using working pattern from single-Tensix version
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, get_compile_time_arg_val(0), tile_size_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, get_compile_time_arg_val(1), tile_size_bytes);

    // Process each A tile with all B tiles
    for (uint32_t i = 0; i < core_num_tiles; i++) {
        // Read A tile for this iteration
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(core_tiles_start + i, a, cb_in0_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        // Read all B tiles for this A tile
        for (uint32_t j = 0; j < r_tiles; j++) {
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);
            noc_async_read_tile(j, b, cb_in1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}
