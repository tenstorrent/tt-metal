// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hw/inc/dataflow_api.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    // Kernel args
    std::uint32_t src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t src_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t num_tiles_r = get_arg_val<uint32_t>(2);
    std::uint32_t num_tiles_c = get_arg_val<uint32_t>(3);

    // How many bytes along a row in the original tensor
    std::uint32_t num_bytes_per_tensor_row = get_arg_val<uint32_t>(4);

    /*
        Constants
        Since I am 'constexpr'ing here, I can multiply
    */
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t num_bytes_per_tile_row = 64;  // 32 bfloat16, each 2 bytes
    constexpr uint32_t num_bytes_for_sending_eight_tile_rows = num_bytes_per_tile_row * 8;
    constexpr uint32_t num_bytes_for_sending_seven_tile_rows = num_bytes_per_tile_row * 7;
    constexpr uint32_t num_bytes_for_sending_twenty_four_tile_rows = num_bytes_per_tile_row * 24;

    // Initialize experimental API objects
    experimental::CircularBuffer cb(cb_id_in0);
    experimental::Noc noc;
    constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
    experimental::AllocatorBank<bank_type> src_dram;

    std::uint32_t num_bytes_per_tile = cb.get_tile_size();

    // Variables
    std::uint32_t start_dram_addr_offset_for_tensor_row = 0;

    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }

    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    for (std::uint32_t i = 0; i < num_tiles_r; i++) {
        for (std::uint32_t j = 0; j < 32; j++) {
            std::uint32_t src_addr_ = src_addr + start_dram_addr_offset_for_tensor_row;
            for (std::uint32_t k = 0; k < num_tiles_c; k++) {
                cb.reserve_back(1);

                // Read one row of data
                std::uint32_t l1_write_addr = cb.get_write_ptr();
                experimental::CoreLocalMem<std::uint32_t> l1_write_buffer(l1_write_addr);

                noc.async_read(
                    src_dram, l1_write_buffer, num_bytes_per_tile_row, {.bank_id = src_bank_id, .addr = src_addr_}, {});

                // We move one row down
                l1_write_addr += num_bytes_per_tile_row;

                /*
                    Move 31 rows of zeros behind the row that we just moved. We send
                    8 rows three times, then we send 7 rows
                    Note: Using old noc API for local L1 to L1 copies via NOC
                */
                for (std::uint32_t z = 0; z < 3; z++) {
                    noc_async_read(zeros_base_noc_addr, l1_write_addr, num_bytes_for_sending_eight_tile_rows);
                    l1_write_addr += num_bytes_for_sending_eight_tile_rows;
                }

                noc_async_read(zeros_base_noc_addr, l1_write_addr, num_bytes_for_sending_seven_tile_rows);

                src_addr_ += num_bytes_per_tile;
                noc.async_read_barrier();
                cb.push_back(1);

            }  // End num_tiles_c loop
            start_dram_addr_offset_for_tensor_row += num_bytes_per_tile_row;
        }  // End 32 iter loop
        start_dram_addr_offset_for_tensor_row += num_bytes_per_tensor_row;
    }  // End num_tiles_r loop
}
