// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_start_tile_id = get_arg(args::in0_tensor_start_tile_id);
    uint32_t in0_tensor_stride_w = get_arg(args::in0_tensor_stride_w);
    uint32_t in0_tensor_stride_h = get_arg(args::in0_tensor_stride_h);
    uint32_t in0_tensor_next_block_stride = get_arg(args::in0_tensor_next_block_stride);

    // in0 block args
    uint32_t in0_block_w = get_arg(args::in0_block_w);
    uint32_t in0_block_h = get_arg(args::in0_block_h);
    uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);

    // in1 tensor args
    uint32_t in1_tensor_start_tile_id = get_arg(args::in1_tensor_start_tile_id);
    uint32_t in1_tensor_stride_w = get_arg(args::in1_tensor_stride_w);
    uint32_t in1_tensor_stride_h = get_arg(args::in1_tensor_stride_h);
    uint32_t in1_tensor_next_block_stride = get_arg(args::in1_tensor_next_block_stride);

    // in1 block args
    uint32_t in1_block_w = get_arg(args::in1_block_w);
    uint32_t in1_block_h = get_arg(args::in1_block_h);
    uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);

    // in0/in1 common args
    uint32_t num_blocks = get_arg(args::num_blocks);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x = get_arg(args::in0_mcast_dest_noc_start_x);
    uint32_t in0_mcast_dest_noc_start_y = get_arg(args::in0_mcast_dest_noc_start_y);
    uint32_t in0_mcast_dest_noc_end_x = get_arg(args::in0_mcast_dest_noc_end_x);
    uint32_t in0_mcast_dest_noc_end_y = get_arg(args::in0_mcast_dest_noc_end_y);
    uint32_t in0_mcast_num_dests = get_arg(args::in0_mcast_num_dests);
    uint32_t in0_mcast_sender_noc_x = get_arg(args::in0_mcast_sender_noc_x);
    uint32_t in0_mcast_sender_noc_y = get_arg(args::in0_mcast_sender_noc_y);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x = get_arg(args::in1_mcast_dest_noc_start_x);
    uint32_t in1_mcast_dest_noc_start_y = get_arg(args::in1_mcast_dest_noc_start_y);
    uint32_t in1_mcast_dest_noc_end_x = get_arg(args::in1_mcast_dest_noc_end_x);
    uint32_t in1_mcast_dest_noc_end_y = get_arg(args::in1_mcast_dest_noc_end_y);
    uint32_t in1_mcast_num_dests = get_arg(args::in1_mcast_num_dests);
    uint32_t in1_mcast_sender_noc_x = get_arg(args::in1_mcast_sender_noc_x);
    uint32_t in1_mcast_sender_noc_y = get_arg(args::in1_mcast_sender_noc_y);

    // batch args
    uint32_t MtKt = get_arg(args::MtKt);  // if 0
    uint32_t KtNt = get_arg(args::KtNt);
    uint32_t batch = get_arg(args::batch);
    uint32_t bcast_B = get_arg(args::bcast_B);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    Semaphore in0_mcast_sender_semaphore(sem::in0_mcast_sender);
    Semaphore in0_mcast_receiver_semaphore(sem::in0_mcast_receiver);
    Semaphore in1_mcast_sender_semaphore(sem::in1_mcast_sender);
    Semaphore in1_mcast_receiver_semaphore(sem::in1_mcast_receiver);
    Noc noc;

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t block = 0; block < num_blocks; block++) {
            // Operand 0
            dfb_in0.reserve_back(in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            in0_mcast_receiver_semaphore.set(INVALID);

            // Atomic increment source core counter
            in0_mcast_sender_semaphore.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);

            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            in0_mcast_receiver_semaphore.wait(VALID);

            dfb_in0.push_back(in0_block_num_tiles);

            // Operand 1
            dfb_in1.reserve_back(in1_block_num_tiles);

            // Set in1 semaphore value to INVALID
            in1_mcast_receiver_semaphore.set(INVALID);

            in1_mcast_sender_semaphore.up(noc, in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            in1_mcast_receiver_semaphore.wait(VALID);

            dfb_in1.push_back(in1_block_num_tiles);
        }
    }
}
