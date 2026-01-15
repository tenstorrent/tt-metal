// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

// This kernel is used to read untilized src0 data from DRAM and copy it to L1 in tilized layout.
// For layout transformation, it uses a list of source addresses (a vector in L1 written by the host) to perform
// scattered and multiple reads from DRAM. The kernel writes to contiguous location in L1 CB. Therefore, the src
// addresses must be provided in the order in which tiles are generated. It expects src1 data to already be tilized and
// it simply copies it to L1.
void kernel_main() {
    std::uint32_t dram_buffer_src0_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src0_bank_id      = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_src1_addr  = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src1_bank_id      = get_arg_val<uint32_t>(3);
    std::uint32_t address_map_size       = get_arg_val<uint32_t>(4);
    std::uint32_t address_map_l1_addr    = get_arg_val<uint32_t>(5);
    std::uint32_t num_blocks             = get_arg_val<uint32_t>(6);
    std::uint32_t src0_num_reads_per_block = get_arg_val<uint32_t>(7);
    std::uint32_t src0_dram_read_size_bytes = get_arg_val<uint32_t>(8);
    std::uint32_t src1_num_bytes_per_block = get_arg_val<uint32_t>(9);
    std::uint32_t src0_num_tiles_per_block = get_arg_val<uint32_t>(10);
    std::uint32_t src1_num_tiles_per_block = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb0_id = 0;
    constexpr uint32_t cb1_id = 1;

    experimental::CircularBuffer cb0(cb0_id);
    experimental::CircularBuffer cb1(cb1_id);
    experimental::CoreLocalMem<std::uint32_t> source_addresses(address_map_l1_addr);
    experimental::Noc noc;

    uint32_t source_addresses_list_index = 0;
    // We push one block of tiles of src0 and src1.
    // src0 and src1 can have different number of tiles per block.
    for (uint32_t b = 0; b < num_blocks; b += 1) {
        cb0.reserve_back(src0_num_tiles_per_block);
        cb1.reserve_back(src1_num_tiles_per_block);

        // src1 is already tilized in DRAM. Read the whole block of tiles in a single DRAM read access.
        // src0 is not tilized in DRAM.
        noc.async_read(
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            cb1,
            src1_num_bytes_per_block,
            {.bank_id = dram_src1_bank_id, .addr = dram_buffer_src1_addr},
            {.offset_bytes = 0});

        // For src0, Do multiple DRAM read accesses using addresses provided in "source_addresses" to produce one block
        // of tiles The source addresses in the list must be in the order of tiles
        for (uint32_t i = 0; i < src0_num_reads_per_block; i++) {
            uint32_t src_addr = source_addresses[source_addresses_list_index];
            noc.async_read(
                experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
                cb0,
                src0_dram_read_size_bytes,
                {.bank_id = dram_src0_bank_id, .addr = dram_buffer_src0_addr + src_addr},
                {.offset_bytes = i * src0_dram_read_size_bytes});
            source_addresses_list_index += 1;
        }
        noc.async_read_barrier();

        dram_buffer_src1_addr += src1_num_bytes_per_block;
        cb0.push_back(src0_num_tiles_per_block);
        cb1.push_back(src1_num_tiles_per_block);
    }
}
