#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    kernel_profiler::mark_time(24);
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
    uint32_t in0_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(34);
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

    volatile uint32_t* in0_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in0_mcast_receiver_semaphore_addr);

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_receiver_semaphore_addr);
    *(in1_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile uint32_t* in1_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile uint32_t*>(in1_mcast_sender_semaphore_addr);

    bool one_time_noc_wait_0 = true;
    bool one_time_noc_wait_1 = true;
    bool one_time_cb_push = true;

    const InterleavedPow2AddrGen s1 = {
        .bank_base_address = in1_tensor_addr,
        .num_used_banks = num_used_dram_ch,
        .log_base_2_of_num_used_banks = num_used_dram_ch_pow2_exponent,
        .log_base_2_of_bank_unit_size = tile_size_pow2_exponent
    };

    for(uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);

        // Set in0 semaphore value to INVALID
        noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

        // Atomic increment source core counter
        uint64_t in0_mcast_sender_semaphore_noc_addr = get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);
        noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
        noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
        kernel_profiler::mark_time_once(25, &one_time_noc_wait_0);

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
                uint64_t in1_tile_noc_address = get_noc_addr(in1_tensor_tile_id, s1);
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

        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value should be in0_mcast_num_dests), then reset
        // the semaphore_addr value back to zero for the next block
        noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, in1_mcast_num_dests);
        noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);
        kernel_profiler::mark_time_once(26, &one_time_noc_wait_1);

        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t in1_multicast_data_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_start_address);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_dests);

        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
        // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

        // We should also multicast the flag to destinations
        uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_receiver_semaphore_addr);
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_semaphore_set_multicast(in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_dests);

        cb_push_back(cb_id_in1, in1_block_num_tiles);
        kernel_profiler::mark_time_once(27, &one_time_cb_push);
    }
    kernel_profiler::mark_time(28);
}
