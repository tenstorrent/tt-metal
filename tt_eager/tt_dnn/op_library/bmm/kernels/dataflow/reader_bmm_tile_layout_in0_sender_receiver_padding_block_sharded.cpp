// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_num_tiles                = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes               = get_compile_time_arg_val(1);
    // in0/in1 common args
    constexpr uint32_t num_blocks                         = get_compile_time_arg_val(2);
    // in0 mcast args
    constexpr uint32_t in0_mcast_sender_semaphore_addr    = get_compile_time_arg_val(3);
    constexpr uint32_t in0_mcast_receiver_semaphore_addr  = get_compile_time_arg_val(4);
    constexpr uint32_t in0_mcast_num_dests                = get_compile_time_arg_val(5);
    constexpr uint32_t in0_mcast_num_cores                = get_compile_time_arg_val(6);
    constexpr uint32_t num_x                              = get_compile_time_arg_val(7);
    constexpr uint32_t num_y                              = get_compile_time_arg_val(8);
    constexpr bool transpose_mcast                        = (bool)get_compile_time_arg_val(9);
    constexpr uint32_t shard_width_in_tiles               = get_compile_time_arg_val(10);
    constexpr uint32_t shard_height_in_tiles              = get_compile_time_arg_val(11);
    constexpr uint32_t in0_block_w                        = get_compile_time_arg_val(12);

    constexpr uint32_t batch                              = get_compile_time_arg_val(13);

    // in1 mcast args
    #ifdef USE_SAME_NOC
    constexpr uint32_t in1_mcast_receiver_semaphore_addr  = get_compile_time_arg_val(14);
    constexpr uint32_t in1_block_num_tiles  = get_compile_time_arg_val(15);
    constexpr uint32_t in1_block_size_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(17);
    #endif

    const uint32_t sender_id                              = get_arg_val<uint32_t>(0);
    const uint32_t in0_mcast_dest_noc_start_x             = get_arg_val<uint32_t>(1);
    const uint32_t in0_mcast_dest_noc_start_y             = get_arg_val<uint32_t>(2);
    const uint32_t in0_mcast_dest_noc_end_x               = get_arg_val<uint32_t>(3);
    const uint32_t in0_mcast_dest_noc_end_y               = get_arg_val<uint32_t>(4);

    // in1 mcast args
    #ifdef USE_SAME_NOC
    const uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(5);
    const uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(6);
    const uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(7);
    const uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(8);
    const uint32_t is_in1_sender                      = get_arg_val<uint32_t>(9);

    volatile tt_l1_ptr uint32_t * in0_mcast_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(10));
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(10 + num_x));
    #else
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5));
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(5 + num_x));
    #endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2; // Sharded cb
    constexpr uint32_t in1_inplace_cb_id = 6;
    constexpr uint32_t cb_sync = 7;

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    // In case we need to send multiple blocks per shard, and shard height in tiles is greater than 1
    // Than we first need to extract the sub-blocks from the shard, and then send them to the destinations
    constexpr bool extract_shard_sub_blocks = shard_height_in_tiles > 1 && num_blocks_per_shard > 1;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    // L1 array
    constexpr uint32_t cb_l1_array = tt::CB::c_in5;
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] = VALID; // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t in0_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);

    uint64_t remote_sender_noc_addrs[num_blocks];
    if constexpr(transpose_mcast) {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_blocks; ++i) {
            remote_sender_noc_addrs[i] = get_noc_addr(in0_mcast_noc_x[x], in0_mcast_noc_y[y], in0_mcast_sender_semaphore_addr);
            ++y;
            if (y == num_y) {
                y = 0;
                ++x;
            }
        }
    } else {
        uint32_t x = 0, y = 0;
        for (uint32_t i = 0; i < num_blocks; ++i) {
            remote_sender_noc_addrs[i] = get_noc_addr(in0_mcast_noc_x[x], in0_mcast_noc_y[y], in0_mcast_sender_semaphore_addr);
            ++x;
            if (x == num_x) {
                x = 0;
                ++y;
            }
        }
    }
    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(
        in0_mcast_dest_noc_start_x,
        in0_mcast_dest_noc_start_y,
        in0_mcast_dest_noc_end_x,
        in0_mcast_dest_noc_end_y,
        0);

    uint64_t in0_mcast_receiver_semaphore_noc_addr = in0_multicast_data_noc | (uint64_t) in0_mcast_receiver_semaphore_addr;

    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, VALID);

    cb_reserve_back(cb_id_in2, batch * in0_block_num_tiles);

    uint32_t local_read_addr = 0;
    uint64_t noc_shard_read_start_addr = 0;
    if constexpr (extract_shard_sub_blocks) {
        noc_shard_read_start_addr = get_noc_addr(get_read_ptr(cb_id_in2));
    } else {
        local_read_addr = get_read_ptr(cb_id_in2);
    }

    #ifdef USE_SAME_NOC
    const uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,

        in1_mcast_receiver_semaphore_addr);

    const uint64_t in1_multicast_data_noc = get_noc_multicast_addr(
        in1_mcast_dest_noc_end_x,
        in1_mcast_dest_noc_end_y,
        in1_mcast_dest_noc_start_x,
        in1_mcast_dest_noc_start_y,
        0);
    #endif

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const uint32_t block_id = block / num_blocks_per_shard;
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            // Set in0 semaphore value to INVALID
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            if (block_id == sender_id) {
                // Operand 0
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

                if constexpr (extract_shard_sub_blocks) {
                    local_read_addr = l1_write_addr_in0;

                    uint32_t l1_write_extract_shard_in0 = l1_write_addr_in0;
                    uint64_t noc_shard_read_addr = noc_shard_read_start_addr;
                    noc_shard_read_start_addr += shard_read_width;

                    for (uint32_t i = 0; i < shard_height_in_tiles; i++) {
                        noc_async_read(noc_shard_read_addr, l1_write_extract_shard_in0, shard_read_width);

                        l1_write_extract_shard_in0 += shard_read_width;
                        noc_shard_read_addr += shard_read_stride;
                    }

                    noc_async_read_barrier();
                }

                // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its value should be in0_mcast_num_dests), then reset
                // the semaphore_addr value back to zero for the next block
                noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                // Now we have the block in the CB address, we can mcast to dests!
                uint64_t in0_multicast_data_addr = in0_multicast_data_noc | l1_write_addr_in0;

                if constexpr (extract_shard_sub_blocks) {
                    // no need to mcast to self, since we are not really doing a local copy
                    noc_async_write_multicast(local_read_addr, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores, false, false);
                } else {
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_async_write_multicast_loopback_src(local_read_addr, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_cores+1, false, false);
                }
                // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
                // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

                // We should also multicast the flag to destinations
                noc_semaphore_set_multicast_loopback_src(in0_mcast_sender_semaphore_valid_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_cores+1, false, false);

                local_read_addr += in0_block_size_bytes;
            } else {
                uint64_t in0_mcast_sender_semaphore_noc_addr = remote_sender_noc_addrs[block_id];

                // Atomic increment source core counter
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            }
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);
            cb_push_back(cb_id_in0, in0_block_num_tiles);

            #ifdef USE_SAME_NOC
            if (is_in1_sender) {
                cb_reserve_back(in1_inplace_cb_id, in1_block_num_tiles);
                cb_wait_front(cb_sync, 1);
                uint32_t in1_start_address = get_write_ptr(in1_inplace_cb_id);
                cb_push_back(in1_inplace_cb_id, in1_block_num_tiles);

                uint64_t in1_multicast_data_addr = in1_multicast_data_noc | in1_start_address;

                noc_async_write_multicast(in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_cores, false, false);

                noc_semaphore_set_multicast(in1_mcast_receiver_semaphore_addr, in1_mcast_receiver_semaphore_noc_addr, in1_mcast_num_cores, false, false);

                cb_pop_front(cb_sync, 1);
            }
            #endif
        }
    }
}
