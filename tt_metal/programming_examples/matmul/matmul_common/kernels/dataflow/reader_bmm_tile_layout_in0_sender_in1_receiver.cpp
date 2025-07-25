// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
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
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(24));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(25));

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(33));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(34));

    // batch args
    uint32_t MtKt = get_arg_val<uint32_t>(35);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(36);
    uint32_t batch = get_arg_val<uint32_t>(37);
    uint32_t bcast_B = get_arg_val<uint32_t>(38);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, in0_tensor_addr, single_tile_size_bytes);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint32_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
            uint32_t in0_block_size_bytes = 0;               // can be optimized later, pass it to kernel

            // Copy in0 block into CB, as the default kernel
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    l1_write_addr_in0 += single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                    in0_block_size_bytes += single_tile_size_bytes;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            // Barrier! make sure the reads are done
            noc_async_read_barrier();

            // wait until all in0 mcast destinations have atomically incremented the in0 semaphore_addr (i.e. its value
            // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
            noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, in0_mcast_num_dests);
            noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);
            // Now we have the block in the CB address, we can mcast to dests!
            uint64_t in0_multicast_data_addr = get_noc_multicast_addr(
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_start_address);
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_async_write_multicast(
                in0_start_address, in0_multicast_data_addr, in0_block_size_bytes, in0_mcast_num_dests);

            // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc even
            // though cmd bufs are different Also, this only works because we are setting VCs statically (using
            // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
            // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not be
            // sent in order they are issued
            noc_async_writes_flushed();
#endif

            // We should also multicast the flag to destinations
            uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
                in0_mcast_dest_noc_end_x,
                in0_mcast_dest_noc_end_y,
                in0_mcast_dest_noc_start_x,
                in0_mcast_dest_noc_start_y,
                in0_mcast_receiver_semaphore_addr);
            // num_dests must not include source, since we are NOT really doing a local copy!
            noc_semaphore_set_multicast(
                in0_mcast_receiver_semaphore_addr, in0_mcast_receiver_semaphore_noc_addr, in0_mcast_num_dests);

            cb_push_back(cb_id_in0, in0_block_num_tiles);

            // Operand 1
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            // Set in1 semaphore value to INVALID
            noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

            uint64_t in1_mcast_sender_semaphore_noc_addr =
                get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
            noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
        in0_tensor_start_tile_id += MtKt;
    }
}
