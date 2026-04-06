// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel with debug checkpoint support.
// Reads tiles from DRAM into input CB, then participates in a checkpoint barrier.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/checkpoint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes = get_tile_size(cb_id) * ublock_size_tiles;

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(src_bank_id, src_addr);
        cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id, ublock_size_tiles);
        src_addr += ublock_size_bytes;
    }

    // Participate in checkpoint barrier at micro-op boundary
    DEBUG_CHECKPOINT(1);
}
