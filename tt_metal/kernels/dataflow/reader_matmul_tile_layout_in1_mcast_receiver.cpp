#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    kernel_profiler::mark_time(16);
    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w                = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h                = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride       = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w                        = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(17);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(18);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(19);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(20);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(21);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(22);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(23);
    uint32_t in1_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(24);
    uint32_t in1_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(25);

    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in1);

    uint32_t l1_write_addr_in0;

    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    bool one_time_noc_wait = true;
    bool one_time_cb_push = true;

    const InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = in0_tensor_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    for(uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);

        uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in0_block_h; h++) {
            uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in0_block_w; w++) {
                uint64_t in0_tile_noc_addr = get_noc_addr(in0_tensor_tile_id, s0);
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

        // Set in0 semaphore value to INVALID
        noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

        uint64_t in1_mcast_sender_semaphore_noc_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
        noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);
        kernel_profiler::mark_time_once(17, &one_time_noc_wait);

        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
        kernel_profiler::mark_time_once(18, &one_time_cb_push);
    }
}
