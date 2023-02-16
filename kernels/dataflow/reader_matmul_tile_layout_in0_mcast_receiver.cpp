#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    kernel_profiler::mark_time(7);
    uint32_t in0_block_num_tiles            = get_arg_val<uint32_t>(0);

    uint32_t in1_tensor_addr                = get_arg_val<uint32_t>(1);
    uint32_t in1_tensor_start_tile_id       = get_arg_val<uint32_t>(2);
    uint32_t in1_tensor_stride_w            = get_arg_val<uint32_t>(3); 
    uint32_t in1_tensor_stride_h            = get_arg_val<uint32_t>(4);
    uint32_t in1_tensor_next_block_stride   = get_arg_val<uint32_t>(5);

    uint32_t in1_block_w                    = get_arg_val<uint32_t>(6);
    uint32_t in1_block_h                    = get_arg_val<uint32_t>(7);
    uint32_t in1_block_num_tiles            = get_arg_val<uint32_t>(8);
    
    uint32_t num_blocks                     = get_arg_val<uint32_t>(9);

    uint32_t in0_mcast_sender_noc_x         = get_arg_val<uint32_t>(10);
    uint32_t in0_mcast_sender_noc_y         = get_arg_val<uint32_t>(11); 

    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in1); 

    uint32_t l1_write_addr_in1;
    
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;

    volatile uint32_t* in0_flag_addr = reinterpret_cast<volatile uint32_t*>(IN0_MCAST_RECEIVER_FLAG);

    bool one_time_noc_wait = true;
    bool one_time_cb_push = true;
    for(uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);

        // Atomic increment source core counter
        uint64_t in0_mcast_sender_semaphore_addr = get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, IN0_MCAST_COUNTER);
        noc_atomic_increment(in0_mcast_sender_semaphore_addr, 1, 31);
        // wait on mcast sender to write value 1 into flag_addr, then reset that flag for the next block

        noc_wait_and_reset(in0_flag_addr, VALID);
        kernel_profiler::mark_time_once(8, &one_time_noc_wait);
        
        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in1_block_w; w++) {
                uint64_t in1_tile_noc_addr = get_noc_addr(in1_tensor_tile_id, in1_tensor_addr, 
                                                num_used_dram_ch, num_used_dram_ch_pow2_exponent, tile_size_pow2_exponent);
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
        kernel_profiler::mark_time_once(9, &one_time_cb_push);
    }
    kernel_profiler::mark_time(10);
}
    

