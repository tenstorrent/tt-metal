// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // READER
    // in1 mcast args
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(0);

    // WRITER
    // out tensor args
    uint32_t out_tensor_addr                    = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_start_tile_id           = get_arg_val<uint32_t>(2);

    // padding args (WRITER)
    uint32_t out_num_nonzero_subblocks_h        = get_arg_val<uint32_t>(3);
    uint32_t out_last_subblock_h                = get_arg_val<uint32_t>(4);
    uint32_t padded_block_tiles_h_skip          = get_arg_val<uint32_t>(5);
    uint32_t out_num_nonzero_subblocks_w        = get_arg_val<uint32_t>(6);
    uint32_t out_last_subblock_w                = get_arg_val<uint32_t>(7);
    uint32_t padded_subblock_tiles_addr_skip    = get_arg_val<uint32_t>(8);
    uint32_t padded_block_tiles_w_skip          = get_arg_val<uint32_t>(9);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr bool out_is_dram                            = get_compile_time_arg_val(0) == 1;

    // READER
    // in1 block args
    constexpr uint32_t in1_block_num_tiles                = get_compile_time_arg_val(1);
    // in0/in1 common args
    constexpr uint32_t num_blocks                         = get_compile_time_arg_val(2);
    // in1 mcast args
    constexpr uint32_t in1_mcast_sender_noc_y             = get_compile_time_arg_val(3);
    uint32_t in1_mcast_sender_semaphore_addr    = get_semaphore(get_compile_time_arg_val(4));
    uint32_t in1_mcast_receiver_semaphore_addr  = get_semaphore(get_compile_time_arg_val(5));
    // batch args
    constexpr uint32_t batch                              = get_compile_time_arg_val(6);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w                = get_compile_time_arg_val(7);
    constexpr uint32_t out_tensor_stride_h                = get_compile_time_arg_val(8);
    constexpr uint32_t out_tensor_next_subblock_stride_w  = get_compile_time_arg_val(9);
    constexpr uint32_t out_tensor_next_subblock_stride_h  = get_compile_time_arg_val(10);

    // out subblock args
    constexpr uint32_t out_subblock_w                     = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_h                     = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_tile_count            = get_compile_time_arg_val(13);

    // batch args
    constexpr uint32_t MtNt                               = get_compile_time_arg_val(14); // if 0
    // Don't need batch; same as batch from READER args

    #ifdef FUSE_BIAS
        // in3 mcast args
        uint32_t in3_mcast_sender_noc_x             = get_arg_val<uint32_t>(10);

        // in3 block args
        constexpr uint32_t in3_block_w                        = get_compile_time_arg_val(15);

        // in3 mcast args
        constexpr uint32_t in3_mcast_sender_noc_y             = get_compile_time_arg_val(16);
        uint32_t in3_mcast_sender_semaphore_addr    = get_semaphore(get_compile_time_arg_val(17));
        uint32_t in3_mcast_receiver_semaphore_addr  = get_semaphore(get_compile_time_arg_val(18));
        constexpr uint32_t cb_id_in3 = 3;
        volatile tt_l1_ptr uint32_t* in3_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in3_mcast_receiver_semaphore_addr);
    #endif

    // WRITER

    constexpr uint32_t cb_id_in1 = 1;

    // WRITER
    constexpr uint32_t cb_id_out0 = 16;

    // WRITER
    // single-tile
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    const DataFormat output_data_format = get_dataformat(cb_id_out0);

    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    // WRITER
    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = out_tensor_addr,
        .page_size = output_single_tile_size_bytes,
        .data_format = output_data_format
    };


    for (uint32_t b = 0; b < batch; b++) {
        for(uint32_t block = 0; block < num_blocks; block++) {
            // Operand 1
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            // Set in1 semaphore value to INVALID
            noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            uint64_t in1_mcast_sender_semaphore_noc_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y, in1_mcast_sender_semaphore_addr);
            noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }

        #ifdef FUSE_BIAS
            // Only read bias on first batch
            if (b == 0) {
                // Operand 2
                cb_reserve_back(cb_id_in3, in3_block_w);

                // Set in1 semaphore value to INVALID
                noc_semaphore_set(in3_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                uint64_t in3_mcast_sender_semaphore_noc_addr = get_noc_addr(in3_mcast_sender_noc_x, in3_mcast_sender_noc_y, in3_mcast_sender_semaphore_addr);
                noc_semaphore_inc(in3_mcast_sender_semaphore_noc_addr, 1);

                // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(in3_mcast_receiver_semaphore_addr_ptr, VALID);

                cb_push_back(cb_id_in3, in3_block_w);
            }
        #endif

        // WRITER
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for(uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for(uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                uint32_t out_subblock_h_ = out_subblock_h;
                uint32_t out_subblock_w_ = out_subblock_w;
                uint32_t subblock_tiles_addr_skip = 0;
                if (sbh == out_num_nonzero_subblocks_h - 1) {
                    out_subblock_h_ = out_last_subblock_h;
                }
                if (sbw == out_num_nonzero_subblocks_w - 1) {
                    out_subblock_w_ = out_last_subblock_w;
                    subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
                }

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for(uint32_t h = 0; h < out_subblock_h_; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for(uint32_t w = 0; w < out_subblock_w_; w++) {
                        noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);

                        l1_read_addr+=output_single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    // Skip padded tiles in subblock along row
                    l1_read_addr += subblock_tiles_addr_skip;
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            // Pop fully padded subblocks along the row
            cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
            cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        // Pop row(s) of fully padded subblocks
        cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
        cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
        out_tensor_start_tile_id += MtNt;
    }
}
