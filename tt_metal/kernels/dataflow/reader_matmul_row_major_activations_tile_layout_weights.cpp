#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {


    bool one_time_profile = true;

    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_row_id            = get_arg_val<uint32_t>(1);

    // in0 block args
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(2);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(3);

    // in0 row size info
    uint32_t in0_row_size                      = get_arg_val<uint32_t>(4); // num bytes in a row
    uint32_t in0_partial_row_size              = get_arg_val<uint32_t>(5); // num bytes in a row that fit within one in0_block_w

    // in1 tensor args (sizes are in num tiles)
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(6);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(7);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(10);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(11);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(12);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(13);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(14);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t in0_tensor_current_block_start_row_id = in0_tensor_start_row_id;
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;

    const InterleavedAddrGen s0 = {
        .bank_base_address = in0_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .bank_unit_size = in0_row_size
    };

    const InterleavedPow2AddrGen s1 = {
        .bank_base_address = in1_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .log_base_2_of_bank_unit_size = tile_size_pow2_exponent
    };

    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE);

    uint32_t row_offset = 0;
    for(uint32_t b = 0; b < num_blocks; b++) {
        uint32_t row_bank_id = in0_tensor_start_row_id;
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);

        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Read in0 row major... will have to read partial rows
        // because the input is row major, but the "block" doesn't
        // cover the full row
        // int some_var = 0;
        for (uint32_t h = 0; h < in0_block_h; h++) {
            for (uint32_t i = 0; i < 32; i++) {
                uint64_t in0_row_noc_addr = get_noc_addr(row_bank_id, s0, row_offset);
                noc_async_read(in0_row_noc_addr, l1_write_addr_in0, in0_partial_row_size);
                l1_write_addr_in0 += in0_partial_row_size;
                row_bank_id++;
            }
            noc_async_read_barrier();
        }
        row_offset = (row_offset + in0_partial_row_size) % in0_row_size;

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
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);

        kernel_profiler::mark_time_once(6, &one_time_profile);
    }
}
