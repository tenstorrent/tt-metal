#include <stdint.h>
#include "dataflow_api.h"
void kernel_main() {
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Arguments for in1
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t in1_block_w = get_arg_val<uint32_t>(2);
    uint32_t in1_block_h = get_arg_val<uint32_t>(3);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(4);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(5);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(6);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(7);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(8);

    // Arguments for in0
    uint32_t src0_addr  = get_arg_val<uint32_t>(9);
    uint32_t in0_block_num_tiles  = get_arg_val<uint32_t>(10);
    uint32_t in0_num_rows = get_arg_val<uint32_t>(11); // in0 has whole column in 1 block
    uint32_t in0_num_channel_sticks_per_row = get_arg_val<uint32_t>(12);
    uint32_t in0_num_channel_sticks_block_w = get_arg_val<uint32_t>(13);
    uint32_t in0_channel_stick_size = get_arg_val<uint32_t>(14);
    uint32_t in0_partial_channel_stick_size = get_arg_val<uint32_t>(15);
    uint32_t num_bytes_of_zeroes_per_read = get_arg_val<uint32_t>(16);
    uint32_t num_reads_of_zeroes = get_arg_val<uint32_t>(17);
    uint32_t num_bytes_of_zeroes_remainder = get_arg_val<uint32_t>(18);
    uint32_t address_map_l1_addr = get_arg_val<uint32_t>(19);
    uint32_t address_map_size = get_arg_val<uint32_t>(20);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    volatile tt_l1_ptr std::uint32_t* channels_address_map = (volatile tt_l1_ptr uint32_t*)(address_map_l1_addr);
    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = src0_addr,


        .page_size = in0_channel_stick_size // size of 1 stick = transfer size in address map = number of conv activation channels
    };

    const InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = src1_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent
    };
    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
    uint32_t row_offset = 0;
    int some_var = 0;
    uint32_t channels_address_map_index = 0;
    uint32_t channel_stick_offset = 0;
    uint32_t channels_address_map_size_per_row = in0_num_channel_sticks_per_row << 2; // 4 entries per channel stick in the address map
    uint32_t in0_num_blocks_per_channel_stick = in0_channel_stick_size / in0_partial_channel_stick_size;
    for(uint32_t b = 0; b < num_blocks; b++) {
        channels_address_map_index = (b / in0_num_blocks_per_channel_stick) << 2; // 4 entries per channel stick in the address map
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);

        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Read weights
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in1_block_w; w++) {
                uint64_t in1_tile_noc_addr = get_noc_addr(in1_tensor_tile_id, s1);
                noc_async_read(in1_tile_noc_addr, l1_write_addr_in1, single_tile_size_bytes);
                l1_write_addr_in1 += single_tile_size_bytes;
                in1_tensor_tile_id += in1_tensor_stride_w;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        noc_async_read_barrier();
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;
        // Read activations using address map
        // Read in0 channels last... will have to read partial sticks
        // because the "block" doesn't cover the full stick
        for (uint32_t h = 0; h < in0_num_rows; h++) {
            uint32_t channels_address_map_start_index = channels_address_map_index;
            //DPRINT << "h= " << h << " channels address map start index= "<< channels_address_map_start_index << ENDL();
            for(uint32_t w = 0; w < in0_num_channel_sticks_block_w; w++) {
                uint32_t src_addr = src0_addr + channels_address_map[channels_address_map_index];
                // Destination address at address_map[am_index+1] unused. Contiguous writes to L1.
                // Transfer size at address_map[am_index+2] unused.
                // Need to do partial transfer because channel stick can be divided between blocks
                uint32_t channel_stick_bank_id = channels_address_map[channels_address_map_index+3];
                uint64_t in0_row_noc_addr = get_noc_addr(channel_stick_bank_id, s0, channel_stick_offset);
                noc_async_read(in0_row_noc_addr, l1_write_addr_in0, in0_partial_channel_stick_size);
                l1_write_addr_in0 += in0_partial_channel_stick_size;
                channels_address_map_index += 4;
            }
            channels_address_map_index = channels_address_map_start_index + channels_address_map_size_per_row;
        }
        // Height padding
        for (uint32_t z = 0; z < num_reads_of_zeroes; z++) {
            noc_async_read(zeros_base_noc_addr, l1_write_addr_in0, num_bytes_of_zeroes_per_read);
            l1_write_addr_in0 += num_bytes_of_zeroes_per_read;
        }
        if(num_bytes_of_zeroes_remainder > 0) {
            noc_async_read(zeros_base_noc_addr, l1_write_addr_in0, num_bytes_of_zeroes_remainder);
            l1_write_addr_in0 += num_bytes_of_zeroes_remainder;
        }

        channel_stick_offset = (channel_stick_offset + in0_partial_channel_stick_size) % in0_channel_stick_size;
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
