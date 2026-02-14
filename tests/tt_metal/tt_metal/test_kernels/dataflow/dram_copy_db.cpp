// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

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

    experimental::CoreLocalMem<uint32_t> l1_mem_1(l1_buffer_addr);
    experimental::CoreLocalMem<uint32_t> l1_mem_2(l1_buffer_addr + rd_wr_l1_buffer_size_bytes);
    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_src_bank;

    // Copy data from DRAM into destination L1 buffer
    noc.async_read(
        dram_src_bank,
        l1_mem_1,
        rd_wr_l1_buffer_size_bytes,
        {.bank_id = dram_src_bank_id, .addr = dram_buffer_src_addr},
        {});

    dram_buffer_src_addr += rd_wr_l1_buffer_size_bytes;
    num_tiles_read += rd_wr_l1_buffer_size_tiles;

    while (num_tiles_read < num_tiles) {
        noc.async_read(
            dram_src_bank,
            l1_mem_2,
            rd_wr_l1_buffer_size_bytes,
            {.bank_id = dram_src_bank_id, .addr = dram_buffer_src_addr},
            {});

        dram_buffer_src_addr += rd_wr_l1_buffer_size_bytes;
        num_tiles_read += rd_wr_l1_buffer_size_tiles;

        // Wait all reads flushed (ie received)
        noc.async_read_barrier();

        noc.async_write(
            l1_mem_1,
            dram_src_bank,
            rd_wr_l1_buffer_size_bytes,
            {},
            {.bank_id = dram_dst_bank_id, .addr = dram_buffer_dst_addr});

        dram_buffer_dst_addr += rd_wr_l1_buffer_size_bytes;

        // Wait for all the writes to complete (ie acked)
        noc.async_write_barrier();

        // Swap L1 addr locations
        if (num_tiles_read < num_tiles) {
            std::swap(l1_mem_1, l1_mem_2);
        }
    }

    // DRAM NOC dst address
    noc.async_write(
        l1_mem_2,
        dram_src_bank,
        rd_wr_l1_buffer_size_bytes,
        {},
        {.bank_id = dram_dst_bank_id, .addr = dram_buffer_dst_addr});
    // Wait for all the writes to complete (ie acked)
    noc.async_write_barrier();
}
