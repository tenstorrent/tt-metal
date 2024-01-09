// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline void noc_async_read_from_dram_to_l1(uint32_t dram_addr, uint32_t dram_noc_x, uint32_t dram_noc_y, uint32_t l1_dest_addr, uint32_t read_size) {
    uint64_t src_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_addr);
    noc_async_read(src_noc_addr, l1_dest_addr, read_size);
}
inline void async_read_from_dram_using_address_map(uint32_t dram_start_addr,
                                            uint32_t dram_noc_x,
                                            uint32_t dram_noc_y,
                                            uint32_t l1_write_addr,
                                            uint32_t address_map_scratch_pad_l1_addr,
                                            uint32_t address_map_group_size,
                                            uint32_t address_map_group_dram_addr,
                                            uint32_t address_map_dram_noc_x,
                                            uint32_t address_map_dram_noc_y) {
    volatile tt_l1_ptr uint32_t * address_map_scratch_pad_buffer = (volatile tt_l1_ptr uint32_t*)(address_map_scratch_pad_l1_addr);
    uint32_t address_map_scratch_pad_buffer_size_bytes = 32; // TODO (nshanker): make this a compile time kernel arg
    uint32_t address_map_scratch_pad_buffer_size = address_map_scratch_pad_buffer_size_bytes >> 2;
    uint32_t address_map_scratch_pad_index = 0;

    for(uint32_t i = 0; i < address_map_group_size; i+=4) {
        if (address_map_scratch_pad_index == 0) {
            // Issue a read from DRAM to fill up the entire scratchpad buffer
            // Scratch pad buffer size must be a multiple of 32B because DRAM read needs to be 32B aligned
            // We want to always do DRAM to L1 write at the start of scratchpad beause l1 write address has to be 32B aligned
            // Kernel assumptions -
            // Host must ensure that "address_map_scratch_pad_l1_addr" % 32 == 0
            // Host must ensure that "address_map_scratch_pad_buffer_size_bytes" % 32 == 0
            noc_async_read_from_dram_to_l1(address_map_group_dram_addr,
            address_map_dram_noc_x, address_map_dram_noc_y,
            address_map_scratch_pad_l1_addr, address_map_scratch_pad_buffer_size_bytes);
            noc_async_read_barrier();
            address_map_group_dram_addr += address_map_scratch_pad_buffer_size_bytes;
        }
        // There are 4 entries in the address map vector for one transfer
        uint32_t src_address_offset = address_map_scratch_pad_buffer[address_map_scratch_pad_index];
        uint32_t dst_address_offset = address_map_scratch_pad_buffer[address_map_scratch_pad_index+1];
        uint32_t read_size = address_map_scratch_pad_buffer[address_map_scratch_pad_index+2];
        uint32_t pad = address_map_scratch_pad_buffer[address_map_scratch_pad_index+3];
        // DPRINT << "src_address_offset=" << src_address_offset << ENDL();
        // DPRINT << "dst_address_offset=" << dst_address_offset << ENDL();
        // DPRINT << "read_size=" << read_size << ENDL();
        // DPRINT << "pad=" << pad << ENDL();

        if(pad == 1) {
            // Insert zeroes in l1
            uint32_t dst_addr = l1_write_addr + dst_address_offset;
            uint32_t pad_size = read_size;
            volatile std::uint8_t* start_dst= (volatile uint8_t*)(dst_addr);
            for (uint32_t offset = 0; offset < pad_size; offset++) {
                *(start_dst + offset) = 0;
            }
            // TODO (nshanker): More performant version below but switched off because it fails non deterministically
            // // source address is set to max. This refers to padding location.
            // // read zeroes from zero buffer
            // uint32_t dst_addr = l1_write_addr + dst_address_offset;
            // uint32_t pad_size = read_size;
            // if (pad_size <= MEM_ZEROS_SIZE) {
            //     noc_async_read(zeros_base_noc_addr, dst_addr, pad_size);
            // }
            // else {
            //     // padding size is bigger than the zero buffer size
            //     // read from zero buffer multiple times
            //     uint32_t zeros_to_read = pad_size;
            //     uint32_t zeros_read_size = MEM_ZEROS_SIZE;
            //     while(zeros_to_read != 0) {
            //         noc_async_read(zeros_base_noc_addr, dst_addr, zeros_read_size);
            //         zeros_to_read -= zeros_read_size;
            //         if (zeros_to_read < zeros_read_size) {
            //             zeros_read_size = zeros_to_read;
            //         }
            //     }
            // }
        }
        else {
            uint32_t src_addr = dram_start_addr + src_address_offset;
            uint32_t dst_addr = l1_write_addr + dst_address_offset;
            noc_async_read_from_dram_to_l1(src_addr, dram_noc_x, dram_noc_y, dst_addr, read_size);
        }
        address_map_scratch_pad_index += 4;
        if(address_map_scratch_pad_index == address_map_scratch_pad_buffer_size) {
            // Reached the end of scratchpad buffer
            // Reset the index to 0 for the next iteration
            address_map_scratch_pad_index = 0;
        }
    }
}
void kernel_main() {
    // Arguments for in0
    uint32_t in0_addr_base  = get_arg_val<uint32_t>(0);
    uint32_t in0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t in0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t in0_address_map_dram_addr = get_arg_val<uint32_t>(3);
    uint32_t in0_address_map_dram_noc_x = get_arg_val<uint32_t>(4);
    uint32_t in0_address_map_dram_noc_y = get_arg_val<uint32_t>(5);
    uint32_t in0_address_map_metadata_l1_addr = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // Arguments for in1
    uint32_t in1_addr_base  = get_arg_val<uint32_t>(8);
    uint32_t in1_noc_x = get_arg_val<uint32_t>(9);
    uint32_t in1_noc_y = get_arg_val<uint32_t>(10);
    uint32_t in1_address_map_dram_addr = get_arg_val<uint32_t>(11);
    uint32_t in1_address_map_dram_noc_x = get_arg_val<uint32_t>(12);
    uint32_t in1_address_map_dram_noc_y = get_arg_val<uint32_t>(13);
     uint32_t in1_address_map_metadata_l1_addr = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    uint32_t scratch_pad_for_address_map_in_l1_addr = get_arg_val<uint32_t>(16);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    // Scratchpad buffer in l1 to stream address map from DRAM into L1
    volatile tt_l1_ptr std::uint32_t* scratch_pad_for_address_map_l1_buffer = (volatile tt_l1_ptr uint32_t*)(scratch_pad_for_address_map_in_l1_addr);
    // Address map metadata buffers in l1. Metadata is copied into L1 buffers by the host before kernel is launched
    volatile tt_l1_ptr std::uint32_t* in0_address_map_metdata_l1_buffer = (volatile tt_l1_ptr uint32_t*)(in0_address_map_metadata_l1_addr);
    volatile tt_l1_ptr std::uint32_t* in1_address_map_metdata_l1_buffer = (volatile tt_l1_ptr uint32_t*)(in1_address_map_metadata_l1_addr);

    // TODO (nshanker): For a more performant padding implementation which is switched off because it fails non deterministically
    // // Put zeroes in the zero buffer for padding
    // constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    // volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    // for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
    //     *(zero_base_ptr + zero_base_offset) = 0;
    // }
    // uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint32_t in0_address_map_metadata_index = 0;
    // address map metdata buffer contains number of groups in the first element
    uint32_t num_groups = in0_address_map_metdata_l1_buffer[in0_address_map_metadata_index];
    in0_address_map_metadata_index += 1;
    // in0 and in1 address maps should have same number of groups
    // no need to get the in1 num of groups
    uint32_t in1_address_map_metadata_index = 1;
    //DPRINT << "num_groups=" << num_groups << ENDL();

    for(uint32_t g = 0; g < num_groups; g++) {

        // Read in0 block from DRAM

        // Read in0 block
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t in0_address_map_current_group_dram_addr_offset = in0_address_map_metdata_l1_buffer[in0_address_map_metadata_index];
        uint32_t in0_address_map_current_group_dram_addr = in0_address_map_dram_addr + in0_address_map_current_group_dram_addr_offset;
        in0_address_map_metadata_index += 1;
        uint32_t in0_address_map_current_group_size = in0_address_map_metdata_l1_buffer[in0_address_map_metadata_index];
        in0_address_map_metadata_index += 1;

        async_read_from_dram_using_address_map(in0_addr_base, in0_noc_x, in0_noc_y,
            l1_write_addr_in0, scratch_pad_for_address_map_in_l1_addr, in0_address_map_current_group_size,
            in0_address_map_current_group_dram_addr, in0_address_map_dram_noc_x, in0_address_map_dram_noc_y);
        noc_async_read_barrier();


        // Read in1 block from DRAM
        // Read in1 block
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        uint32_t in1_address_map_current_group_dram_addr_offset = in1_address_map_metdata_l1_buffer[in1_address_map_metadata_index];
        uint32_t in1_address_map_current_group_dram_addr = in1_address_map_dram_addr + in1_address_map_current_group_dram_addr_offset;
        in1_address_map_metadata_index += 1;
        uint32_t in1_address_map_current_group_size = in1_address_map_metdata_l1_buffer[in1_address_map_metadata_index];
        in1_address_map_metadata_index += 1;
        async_read_from_dram_using_address_map(in1_addr_base, in1_noc_x, in1_noc_y,
            l1_write_addr_in1, scratch_pad_for_address_map_in_l1_addr, in1_address_map_current_group_size,
            in1_address_map_current_group_dram_addr, in1_address_map_dram_noc_x, in1_address_map_dram_noc_y);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
