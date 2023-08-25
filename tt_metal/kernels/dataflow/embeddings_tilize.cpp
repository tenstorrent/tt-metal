// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"


#define CB_ID 0

void kernel_main() {
    std::uint32_t input_offset = get_arg_val<uint32_t>(0);
    std::uint32_t num_blocks      = get_arg_val<uint32_t>(1);
    std::uint32_t input_dram_buffer_src_addr  = get_arg_val<uint32_t>(2);
    std::uint32_t weights_dram_buffer_src_addr  = get_arg_val<uint32_t>(3);


    #define in_is_dram get_compile_time_arg_val(0) == 1
    #define weights_is_dram get_compile_time_arg_val(1) == 1
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t input_l1_addr      = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_block      = get_compile_time_arg_val(4);
    #define weight_stick_size_is_power_of_two get_compile_time_arg_val(5) == 1


    const InterleavedAddrGen<in_is_dram> input = {
        .bank_base_address = input_dram_buffer_src_addr, .page_size = sizeof(uint32_t)};

    #if (weight_stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(6);
    const InterleavedPow2AddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr,
        .page_size = stick_size
    };
    #endif


    constexpr uint32_t tile_height = 32;

    uint32_t index = 0;

    uint64_t base_src_noc_addr[tile_height];


    auto read_tiles = [&] (const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_reserve_back(CB_ID, num_tiles);
        uint32_t l1_write_addr = get_write_ptr(CB_ID);
        for (uint32_t k = 0; k < tile_height; k++) {
            uint64_t src_noc_addr = base_src_noc_addr[k];
            noc_async_read(src_noc_addr, l1_write_addr, width_size);
            l1_write_addr += width_size;
        }


        noc_async_read_barrier();
        cb_push_back(CB_ID, num_tiles);
    };


    for (uint32_t i = 0; i < num_blocks; i++) {
        for(uint32_t j=0; j < tile_height; j++){
            auto noc_input_src_addr = get_noc_addr(j+input_offset+index, input);
            noc_async_read(noc_input_src_addr, input_l1_addr, sizeof(uint32_t));
            noc_async_read_barrier();

            uint32_t row = ((uint32_t *)input_l1_addr)[0];
            auto noc_src_addr = get_noc_addr(row, weights);
            base_src_noc_addr[j] = noc_src_addr;
        }
        index+=tile_height;
        read_tiles(tiles_per_block, stick_size);
    }

}
