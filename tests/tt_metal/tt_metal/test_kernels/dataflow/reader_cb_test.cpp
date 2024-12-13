// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline __attribute__((always_inline))
void read_and_push_to_cb(const uint32_t cb_id, uint32_t num_tiles_per_cb, uint32_t ublock_size_tiles, uint32_t ublock_size_bytes,
                               uint32_t bank_id, uint32_t& dram_buffer_src_addr) {
    // read a ublock of tiles at the time from DRAM to L1 buffer, and push a ublock at the time to unpacker
    for (uint32_t i = 0; i < num_tiles_per_cb; i += ublock_size_tiles) {
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dram_buffer_src_addr);
        cb_reserve_back(cb_id, ublock_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        noc_async_read(dram_buffer_src_noc_addr, l1_write_addr, ublock_size_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id, ublock_size_tiles);
        dram_buffer_src_addr += ublock_size_bytes;
    }
}

void kernel_main() {
    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t bank_id               = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles_per_cb      = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t ublock_size_tiles = get_compile_time_arg_val(1);
    uint32_t ublock_size_bytes = get_tile_size(cb_id) * ublock_size_tiles;

    read_and_push_to_cb(cb_id, num_tiles_per_cb, ublock_size_tiles, ublock_size_bytes,
                              bank_id, dram_buffer_src_addr);
}
