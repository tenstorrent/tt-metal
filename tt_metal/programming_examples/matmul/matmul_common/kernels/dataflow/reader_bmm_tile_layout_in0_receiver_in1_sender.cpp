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

    const uint32_t single_tile_size_bytes = dfb_in1.get_entry_size();

    uint32_t l1_write_addr_in1;

    Semaphore in0_mcast_sender_semaphore(sem::in0_mcast_sender);
    Semaphore in0_mcast_receiver_semaphore(sem::in0_mcast_receiver);
    Noc noc;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    Semaphore in1_mcast_receiver_semaphore(sem::in1_mcast_receiver);
    in1_mcast_receiver_semaphore.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    Semaphore in1_mcast_sender_semaphore(sem::in1_mcast_sender);

    // The in1 tensor is bound to this kernel by name in the host code.
    const auto s1 = TensorAccessor(tensor::in1);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
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
            l1_write_addr_in1 = dfb_in1.get_write_ptr();

            uint32_t in1_start_address = l1_write_addr_in1;  // copy start address of block, to be used for mcasting
            uint32_t in1_block_size_bytes = 0;               // can be optimized later, pass it to kernel

            // Copy in1 block into DFB, as the default kernel
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_page(in1_tensor_tile_id, s1, l1_write_addr_in1);
                    l1_write_addr_in1 += single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                    in1_block_size_bytes += single_tile_size_bytes;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();

            // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value
            // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
            in1_mcast_sender_semaphore.wait(in1_mcast_num_dests);
            in1_mcast_sender_semaphore.set(0);

            // Now we have the block in the DFB address, we can mcast to dests!
            uint64_t in1_multicast_data_addr = get_noc_multicast_addr(
                in1_mcast_dest_noc_start_x,
                in1_mcast_dest_noc_start_y,
                in1_mcast_dest_noc_end_x,
                in1_mcast_dest_noc_end_y,
                in1_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                in1_start_address, in1_multicast_data_addr, in1_block_size_bytes, in1_mcast_num_dests);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc even
            // though cmd bufs are different Also, this only works because we are setting VCs statically (using
            // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
            // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not be
            // sent in order they are issued
            noc_async_writes_flushed();
#endif

            // We should also multicast the flag to destinations
            // num_dests must not include source, since we are NOT really doing a local copy!
            in1_mcast_receiver_semaphore.set_multicast(
                noc,
                in1_mcast_dest_noc_start_x,
                in1_mcast_dest_noc_start_y,
                in1_mcast_dest_noc_end_x,
                in1_mcast_dest_noc_end_y,
                in1_mcast_num_dests);

            dfb_in1.push_back(in1_block_num_tiles);
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
    }
}
