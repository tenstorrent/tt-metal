// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t dram_buffer_src0_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src0_bank_id = get_arg_val<uint32_t>(1);

    std::uint32_t dram_buffer_src1_addr = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src1_bank_id = get_arg_val<uint32_t>(3);

    std::uint32_t num_tiles = get_arg_val<uint32_t>(4);

    // single-tile chunks
    uint32_t chunk_size_bytes_0 = get_tile_size(0);
    uint32_t chunk_size_bytes_1 = get_tile_size(1);
    uint32_t chunk_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // read a chunk of tiles at the time from DRAM to L1 buffer, and push a chunk at the time to unpacker
    for (uint32_t i = 0; i < num_tiles; i += chunk_size_tiles) {
        // DRAM NOC src address
        std::uint64_t dram_buffer_src0_noc_addr =
            get_noc_addr_from_bank_id<true>(dram_src0_bank_id, dram_buffer_src0_addr);
        std::uint64_t dram_buffer_src1_noc_addr =
            get_noc_addr_from_bank_id<true>(dram_src1_bank_id, dram_buffer_src1_addr);

        cb_reserve_back(0, chunk_size_tiles);
        cb_reserve_back(1, chunk_size_tiles);
        l1_write_addr_in0 = get_write_ptr(0);
        l1_write_addr_in1 = get_write_ptr(1);

        noc_async_read(dram_buffer_src0_noc_addr, l1_write_addr_in0, chunk_size_bytes_0);
        noc_async_read(dram_buffer_src1_noc_addr, l1_write_addr_in1, chunk_size_bytes_1);

        noc_async_read_barrier();

        cb_push_back(0, chunk_size_tiles);
        cb_push_back(1, chunk_size_tiles);

        dram_buffer_src0_addr += chunk_size_bytes_0;
        dram_buffer_src1_addr += chunk_size_bytes_1;
    }
}
