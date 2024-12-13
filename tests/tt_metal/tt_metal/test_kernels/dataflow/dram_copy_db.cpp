// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include "dataflow_api.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base   = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_bank_id            = get_arg_val<uint32_t>(1);

    std::uint32_t dram_buffer_dst_addr_base   = get_arg_val<uint32_t>(2);
    std::uint32_t dram_dst_bank_id            = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_size            = get_arg_val<uint32_t>(4);
    std::uint32_t num_tiles                   = get_arg_val<uint32_t>(5);

    std::uint32_t l1_buffer_addr              = get_arg_val<uint32_t>(6);
    std::uint32_t l1_buffer_size_tiles        = get_arg_val<uint32_t>(7);
    std::uint32_t l1_buffer_size_bytes        = get_arg_val<uint32_t>(8);

    std::uint32_t rd_wr_l1_buffer_size_tiles = l1_buffer_size_tiles / 2;
    std::uint32_t rd_wr_l1_buffer_size_bytes = l1_buffer_size_bytes / 2;

    // Keeps track of how many tiles we copied so far
    std::uint32_t num_tiles_read = 0;

    std::uint32_t dram_buffer_src_addr = dram_buffer_src_addr_base;
    std::uint32_t dram_buffer_dst_addr = dram_buffer_dst_addr_base;
    std::uint64_t dram_buffer_src_noc_addr;
    std::uint64_t dram_buffer_dst_noc_addr;

    std::uint32_t l1_addr1 = l1_buffer_addr;
    std::uint32_t l1_addr2 = l1_buffer_addr + rd_wr_l1_buffer_size_bytes;

    // DRAM NOC src address
    dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(dram_src_bank_id, dram_buffer_src_addr);

    // Copy data from DRAM into destination L1 buffer
    noc_async_read(dram_buffer_src_noc_addr, l1_addr1, rd_wr_l1_buffer_size_bytes);
    dram_buffer_src_addr += rd_wr_l1_buffer_size_bytes;
    num_tiles_read += rd_wr_l1_buffer_size_tiles;

    while (num_tiles_read < num_tiles) {
        // DRAM NOC src address
        dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(dram_src_bank_id, dram_buffer_src_addr);
        // DRAM NOC dst address
        dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_dst_bank_id, dram_buffer_dst_addr);

        noc_async_read(dram_buffer_src_noc_addr, l1_addr2, rd_wr_l1_buffer_size_bytes);
        dram_buffer_src_addr += rd_wr_l1_buffer_size_bytes;
        num_tiles_read += rd_wr_l1_buffer_size_tiles;

        // Wait all reads flushed (ie received)
        noc_async_read_barrier();

        noc_async_write(l1_addr1, dram_buffer_dst_noc_addr, rd_wr_l1_buffer_size_bytes);

        dram_buffer_dst_addr += rd_wr_l1_buffer_size_bytes;

        // Wait for all the writes to complete (ie acked)
        noc_async_write_barrier();

        // Swap L1 addr locations
        if (num_tiles_read < num_tiles) {
            std::uint32_t temp_l1_addr = l1_addr1;
            l1_addr1 = l1_addr2;
            l1_addr2 = temp_l1_addr;
        }
    }

    // DRAM NOC dst address
    dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_dst_bank_id, dram_buffer_dst_addr);
    noc_async_write(l1_addr2, dram_buffer_dst_noc_addr, rd_wr_l1_buffer_size_bytes);
    // Wait for all the writes to complete (ie acked)
    noc_async_write_barrier();
}
