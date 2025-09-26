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
    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(2);
    uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(3);

    // padding args
    uint32_t last_block_h = get_arg_val<uint32_t>(4);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr auto in0_args = TensorAccessorArgs<0>();

    // in0 tensor args
    constexpr uint32_t in0_tensor_stride_w = get_compile_time_arg_val(1);
    constexpr uint32_t in0_tensor_stride_h = get_compile_time_arg_val(2);
    constexpr uint32_t in0_tensor_next_block_stride = get_compile_time_arg_val(3);
    // in0 block args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    // in0 mcast args
    constexpr uint32_t in0_mcast_dest_noc_start_x = get_compile_time_arg_val(8);
    constexpr uint32_t in0_mcast_dest_noc_end_x = get_compile_time_arg_val(9);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(12);
    // batch args
    constexpr uint32_t MtKt = get_compile_time_arg_val(13);  // if 0
    constexpr uint32_t batch = get_compile_time_arg_val(14);

    constexpr uint32_t cb_id_in0 = 0;

    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_write_addr_in0;

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            // Operand 0
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint32_t in0_start_address = l1_write_addr_in0;  // copy start address of block, to be used for mcasting
            uint32_t in0_block_size_bytes = 0;               // can be optimized later, pass it to kernel

            // Copy in0 block into CB, as the default kernel
            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    if (h < last_block_h) {
                        noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
                    }
                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                    in0_block_size_bytes += in0_single_tile_size_bytes;
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
        }
        in0_tensor_start_tile_id += MtKt;
    }
}
