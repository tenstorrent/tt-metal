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
    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    experimental::Noc noc;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in1(cb_id_in1);

    uint32_t single_tile_size_bytes = cb_in0.get_tile_size();

    uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;

    experimental::Semaphore in0_mcast_sender_semaphore(in0_mcast_sender_semaphore_id);
    experimental::Semaphore in0_mcast_receiver_semaphore(in0_mcast_receiver_semaphore_id);
    experimental::Semaphore in1_mcast_sender_semaphore(in1_mcast_sender_semaphore_id);
    experimental::Semaphore in1_mcast_receiver_semaphore(in1_mcast_receiver_semaphore_id);

    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in1_mcast_receiver_semaphore.set(VALID);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, single_tile_size_bytes);

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

        uint32_t in1_block_size_bytes = 0;               // can be optimized later, pass it to kernel

        // Copy in1 block into CB, as the default kernel
        uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in1_block_h; h++) {
            uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
            for (uint32_t w = 0; w < in1_block_w; w++) {
                noc.async_read(
                    s1,
                    cb_in1,
                    single_tile_size_bytes,
                    {.page_id = in1_tensor_tile_id},
                    {.offset_bytes = ((h * in1_block_w + w) * single_tile_size_bytes)});
                in1_tensor_tile_id += in1_tensor_stride_w;
                in1_block_size_bytes += single_tile_size_bytes;
            }
            in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

        // Barrier! make sure the reads are done
        noc.async_read_barrier();

        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value
        // should be in0_mcast_num_dests), then reset the semaphore_addr value back to zero for the next block
        in1_mcast_sender_semaphore.down(in1_mcast_num_dests);

        // Now we have the block in the CB address, we can mcast to dests!
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc.async_write_multicast(
            experimental::use<experimental::CircularBuffer::AddrSel::WRITE_PTR>(cb_in1),
            experimental::use<experimental::CircularBuffer::AddrSel::WRITE_PTR>(cb_in1),
            in1_block_size_bytes,
            in1_mcast_num_dests,
            {},
            {.noc_x_start = in1_mcast_dest_noc_start_x,
             .noc_y_start = in1_mcast_dest_noc_start_y,
             .noc_x_end = in1_mcast_dest_noc_end_x,
             .noc_y_end = in1_mcast_dest_noc_end_y});

        // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc even
        // though cmd bufs are different Also, this only works because we are setting VCs statically (using
        // NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
        // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may not be sent
        // in order they are issued
        noc.async_writes_flushed();
#endif

        // We should also multicast the flag to destinations
        // num_dests must not include source, since we are NOT really doing a local copy!
        in1_mcast_receiver_semaphore.set_multicast(
            noc,
            in1_mcast_dest_noc_start_x,
            in1_mcast_dest_noc_start_y,
            in1_mcast_dest_noc_end_x,
            in1_mcast_dest_noc_end_y,
            in1_mcast_num_dests
        );

        cb_in1.push_back(in1_block_num_tiles);
    }
}
