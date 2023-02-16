#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    uint32_t in0_tensor_addr                = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id       = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w            = get_arg_val<uint32_t>(2); 
    uint32_t in0_tensor_stride_h            = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride   = get_arg_val<uint32_t>(4);

    uint32_t in0_block_w                    = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                    = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles            = get_arg_val<uint32_t>(7);
    
    uint32_t in1_block_num_tiles            = get_arg_val<uint32_t>(8);
    
    uint32_t num_blocks                     = get_arg_val<uint32_t>(9);

    uint32_t in1_mcast_sender_noc_x         = get_arg_val<uint32_t>(10);
    uint32_t in1_mcast_sender_noc_y         = get_arg_val<uint32_t>(11); 

    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in1); 

    uint32_t l1_write_addr_in0;
    
    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    volatile uint32_t* in1_flag_addr = reinterpret_cast<volatile uint32_t*>(IN1_MCAST_RECEIVER_FLAG);
    for(uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);

        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in0_block_h; h++) {
            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in0_block_w; w++) {
                uint64_t in0_tile_noc_addr = get_noc_addr(in0_tensor_tile_id, in0_tensor_addr, 
                                                num_used_dram_ch, num_used_dram_ch_pow2_exponent, tile_size_pow2_exponent);
                noc_async_read(in0_tile_noc_addr, l1_write_addr_in0, single_tile_size_bytes);
                l1_write_addr_in0 += single_tile_size_bytes;
                in0_tensor_tile_id += in0_tensor_stride_w;
            }
            in0_tensor_row_start_tile_id += in0_tensor_stride_h;
        }
        in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

        noc_async_read_barrier();
        
        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        // Atomic increment source core counter
        *(in1_flag_addr) = INVALID;
        uint64_t in1_mcast_sender_semaphore_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, IN1_MCAST_COUNTER);
        noc_atomic_increment(in1_mcast_sender_semaphore_addr, 1, 31);
        // wait on mcast sender to write value 1 into flag_addr, then reset that flag for the next block
        noc_wait_and_reset(in1_flag_addr, VALID);

        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
    

