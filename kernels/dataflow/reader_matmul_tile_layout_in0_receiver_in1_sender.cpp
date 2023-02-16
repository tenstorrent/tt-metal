#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
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
    uint32_t in0_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests                = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x             = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y             = get_arg_val<uint32_t>(23); 

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(24);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(25);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(30); 

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in1;
    
    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;

    volatile uint32_t* in0_flag_addr = reinterpret_cast<volatile uint32_t*>(IN0_MCAST_RECEIVER_FLAG);

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in1_mcast_destination_flag_addr = reinterpret_cast<volatile uint32_t*>(IN1_MCAST_RECEIVER_FLAG);
    *(in1_mcast_destination_flag_addr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile uint32_t* in1_semaphore_addr = reinterpret_cast<volatile uint32_t*>(IN1_MCAST_COUNTER);
    
    for(uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);

        // Atomic increment source core counter
        *(in0_flag_addr) = INVALID;
        uint64_t in0_mcast_sender_semaphore_addr = get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, IN0_MCAST_COUNTER);
        noc_atomic_increment(in0_mcast_sender_semaphore_addr, 1, 31);
        // wait on mcast sender to write value 1 into flag_addr, then reset that flag for the next block
        noc_wait_and_reset(in0_flag_addr, VALID);
        
        cb_push_back(cb_id_in0, in0_block_num_tiles);

        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        uint32_t in1_start_address = l1_write_addr_in1; // copy start address of block, to be used for mcasting
        uint32_t in1_block_size_bytes = 0; // can be optimized later, pass it to kernel

        // Copy in1 block into CB, as the default kernel
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for(uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for(uint32_t w = 0; w < in1_block_w; w++) {
                uint64_t in1_tile_noc_address = get_noc_addr(in1_tensor_tile_id, in1_tensor_addr, 
                                                num_used_dram_ch, num_used_dram_ch_pow2_exponent, tile_size_pow2_exponent);
                noc_async_read(in1_tile_noc_address, l1_write_addr_in1, single_tile_size_bytes);
                l1_write_addr_in1 += single_tile_size_bytes;
                in1_tensor_tile_id += in1_tensor_stride_w;
                in1_block_size_bytes += single_tile_size_bytes;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride; 

        // Barrier! make sure the reads are done
        noc_async_read_barrier();

        // wait until all mcast destinations have atomically incremented the semaphore_addr (i.e. its value should be in1_mcast_num_dests), then reset
        // the semaphore_addr value back to zero for the next block
        noc_wait_and_reset(in1_semaphore_addr, in1_mcast_num_dests);
        
        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t in1_multicast_data_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, 
        in1_mcast_dest_noc_start_y, 
        in1_mcast_dest_noc_end_x, 
        in1_mcast_dest_noc_end_y, 
        in1_start_address);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_dests);
        noc_async_write_barrier();
        // We should also multicast the flag to destinations
        uint64_t in1_multicast_flag_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x, 
        in1_mcast_dest_noc_start_y, 
        in1_mcast_dest_noc_end_x, 
        in1_mcast_dest_noc_end_y, 
        IN1_MCAST_RECEIVER_FLAG);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(IN1_MCAST_RECEIVER_FLAG, in1_multicast_flag_addr, 4, in1_mcast_num_dests);
        noc_async_write_barrier();

        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
    

