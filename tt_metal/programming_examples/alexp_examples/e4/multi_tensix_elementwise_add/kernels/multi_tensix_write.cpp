// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t c_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);  // Number of tiles for this Tensix core
    uint32_t tile_offset = get_arg_val<uint32_t>(2);  // Starting tile offset for this core

    constexpr uint32_t cb_out = tt::CB::c_out0;

    const uint32_t tile_size_bytes = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> c(c_addr, tile_size_bytes, tile_offset);

    // Write result tiles from this Tensix core
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, c, cb_out_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
