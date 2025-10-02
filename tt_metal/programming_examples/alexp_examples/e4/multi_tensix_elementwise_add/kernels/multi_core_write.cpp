// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Runtime arguments
    uint32_t core_tiles_start = get_arg_val<uint32_t>(0);  // Starting tile index for this core
    uint32_t core_num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles for this core

    constexpr uint32_t cb_out = tt::CB::c_out0;

    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    // Create tensor accessor using working pattern from single-Tensix version
    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, get_compile_time_arg_val(0), tile_size_bytes);

    // Write result tiles from this Tensix core
    for (uint32_t i = 0; i < core_num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_out);
        noc_async_write_tile(core_tiles_start + i, c, cb_out_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
