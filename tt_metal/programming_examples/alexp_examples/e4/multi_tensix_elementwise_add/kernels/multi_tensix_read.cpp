// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);  // Number of A tiles for this Tensix core
    uint32_t r_tiles = get_arg_val<uint32_t>(3);  // Number of B tiles (replicated)
    uint32_t tile_offset = get_arg_val<uint32_t>(4);  // Starting tile offset for this core

    constexpr uint32_t cb_in0 = tt::CB::c_in0;  // A tiles (distributed)
    constexpr uint32_t cb_in1 = tt::CB::c_in1;  // B tiles (replicated)

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create tensor accessors
    const InterleavedAddrGenFast<true> a(a_addr, tile_size_bytes, tile_offset);
    const InterleavedAddrGenFast<true> b(b_addr, tile_size_bytes);

    // Read A tiles assigned to this Tensix core
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(i, a, cb_in0_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        // For each A tile, read all B tiles (replicated across all cores)
        for (uint32_t j = 0; j < r_tiles; j++) {
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);
            noc_async_read_tile(j, b, cb_in1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}
