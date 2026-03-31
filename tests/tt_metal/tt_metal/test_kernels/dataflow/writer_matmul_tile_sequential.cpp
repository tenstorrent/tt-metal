// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Simple sequential writer for tile-by-tile matmul tests.
// Reads tiles one at a time from CB16 and writes them sequentially to a
// single-bank DRAM buffer.
//
// Runtime args:
//   [0] dst_addr  — DRAM buffer address
//   [1] bank_id   — DRAM bank ID
//   [2] num_tiles — total output tiles to write

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_addr = get_read_ptr(cb_id_out);
        uint64_t noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
        noc_async_write(l1_addr, noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
        dst_addr += tile_bytes;
    }
}
