// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t batch_offset = get_arg_val<uint32_t>(0);
    std::uint32_t weights_offset = get_arg_val<uint32_t>(1);
    std::uint32_t num_blocks      = get_arg_val<uint32_t>(2);
    std::uint32_t input_dram_buffer_src_addr  = get_arg_val<uint32_t>(3);
    std::uint32_t weights_dram_buffer_src_addr  = get_arg_val<uint32_t>(4);


    #define in_is_dram get_compile_time_arg_val(0) == 1
    #define in_stick_size_is_power_of_two get_compile_time_arg_val(1) == 1
    constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
    #if (in_stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_input_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<in_is_dram> input = {
        .bank_base_address = input_dram_buffer_src_addr,
        .log_base_2_of_page_size = log_base_2_of_input_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<in_is_dram> input = {
        .bank_base_address = input_dram_buffer_src_addr,
        .page_size = input_page_size
    };
    #endif

    #define weights_is_dram get_compile_time_arg_val(4) == 1
    #define weight_stick_size_is_power_of_two get_compile_time_arg_val(5) == 1
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(6);
    #if (weight_stick_size_is_power_of_two)
    constexpr uint32_t log_base_2_of_weights_page_size = get_compile_time_arg_val(7);
    const InterleavedPow2AddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr,
        .log_base_2_of_page_size = log_base_2_of_weights_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<weights_is_dram> weights = {
        .bank_base_address = weights_dram_buffer_src_addr,
        .page_size = weight_stick_size
    };
    #endif
    constexpr uint32_t tiles_per_block             = get_compile_time_arg_val(8);
    constexpr uint32_t input_block_size_bytes      = get_compile_time_arg_val(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr uint32_t tile_height = 32;

    uint64_t base_src_noc_addr[tile_height];

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);
    volatile tt_l1_ptr uint32_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
    auto read_tiles = [&input_l1_ptr, &weights] (const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_reserve_back(cb_id_in0, num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = 0; k < tile_height; k++) {
            uint64_t src_noc_addr = get_noc_addr(input_l1_ptr[k], weights);
            noc_async_read(src_noc_addr, l1_write_addr, width_size);
            l1_write_addr += width_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_tiles);
    };

    uint32_t curr_row = batch_offset;
    uint32_t offset = weights_offset;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint64_t noc_input_src_addr = get_noc_addr(curr_row, input) + offset;
        noc_async_read(noc_input_src_addr, input_l1_addr, input_block_size_bytes);
        noc_async_read_barrier();
        read_tiles(tiles_per_block, weight_stick_size);
        offset += input_block_size_bytes;
        if (offset == input_page_size) {
            offset = 0;
            curr_row++;
        }
    }
}
