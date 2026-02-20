// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_id = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_id = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_id = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_id = get_arg_val<uint32_t>(34);

    experimental::Noc noc;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);

    experimental::Semaphore in0_mcast_sender_semaphore(in0_mcast_sender_semaphore_id);
    experimental::Semaphore in0_mcast_receiver_semaphore(in0_mcast_receiver_semaphore_id);
    experimental::Semaphore in1_mcast_sender_semaphore(in1_mcast_sender_semaphore_id);
    experimental::Semaphore in1_mcast_receiver_semaphore(in1_mcast_receiver_semaphore_id);

    for (uint32_t b = 0; b < num_blocks; b++) {
        // Operand 0
        cb_in0.reserve_back(in0_block_num_tiles);

        // Set in0 semaphore value to INVALID
        in0_mcast_receiver_semaphore.set(INVALID);

        // Atomic increment source core counter
        in0_mcast_sender_semaphore.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);

        // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
        in0_mcast_receiver_semaphore.wait(VALID);

        cb_in0.push_back(in0_block_num_tiles);

        // Operand 1
        cb_in1.reserve_back(in1_block_num_tiles);

        // Set in1 semaphore value to INVALID
        in1_mcast_receiver_semaphore.set(INVALID);

        in1_mcast_sender_semaphore.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

        // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
        in1_mcast_receiver_semaphore.wait(VALID);

        cb_in1.push_back(in1_block_num_tiles);
    }
}
