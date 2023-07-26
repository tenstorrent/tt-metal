#include <cstdlib>
#include "dataflow_api.h"

void kernel_main() {
    // Kernel args
    uint32_t src_addr                      = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x                     = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y                     = get_arg_val<uint32_t>(2);
    uint32_t num_tiles_r                   = get_arg_val<uint32_t>(3);
    uint32_t num_tiles_c                   = get_arg_val<uint32_t>(4);

    // How many bytes along a row in the original tensor
    uint32_t num_bytes_per_tensor_row      = get_arg_val<uint32_t>(5);

    /*
        Constants
        Since I am 'constexpr'ing here, I can multiply
    */
    constexpr uint32_t cb_id_in0                                    = 0;
    constexpr uint32_t num_bytes_per_tile_row                       = 64; // 32 bfloat16, each 2 bytes
    constexpr uint32_t num_bytes_for_sending_eight_tile_rows        = num_bytes_per_tile_row * 8;
    constexpr uint32_t num_bytes_for_sending_seven_tile_rows        = num_bytes_per_tile_row * 7;
    constexpr uint32_t num_bytes_for_sending_twenty_four_tile_rows  = num_bytes_per_tile_row * 24;
    uint32_t num_bytes_per_tile                                     = get_tile_size(cb_id_in0);

    // Variables
    uint64_t replicate_dest_addr;
    uint32_t start_dram_addr_offset_for_tensor_row = 0;

    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }

    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t i = 0; i < num_tiles_r; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            uint32_t src_addr_ = src_addr + start_dram_addr_offset_for_tensor_row;
            for (uint32_t k = 0; k < num_tiles_c; k++) {
                cb_reserve_back(cb_id_in0, 1);
                uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr_);

                // Read one row of data
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read(src_noc_addr, l1_write_addr, num_bytes_per_tile_row);

                // We move one row down
                l1_write_addr += num_bytes_per_tile_row;

                /*
                    Move 31 rows of zeros behind the row that we just moved. We send
                    8 rows three times, then we send 7 rows
                */
                for (uint32_t z = 0; z < 3; z++) {
                    noc_async_read(zeros_base_noc_addr, l1_write_addr, num_bytes_for_sending_eight_tile_rows);
                    l1_write_addr += num_bytes_for_sending_eight_tile_rows;
                }

                noc_async_read(zeros_base_noc_addr, l1_write_addr, num_bytes_for_sending_seven_tile_rows);

                src_addr_ += num_bytes_per_tile;
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);

            } // End num_tiles_c loop
            start_dram_addr_offset_for_tensor_row += num_bytes_per_tile_row;
        } // End 32 iter loop
        start_dram_addr_offset_for_tensor_row += num_bytes_per_tensor_row;
    } // End num_tiles_r loop
}
