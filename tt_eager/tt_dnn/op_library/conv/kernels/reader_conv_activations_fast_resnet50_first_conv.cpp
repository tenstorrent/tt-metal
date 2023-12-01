// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"

void kernel_main() {
    uint32_t i = 0;
    uint32_t act_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;

    uint32_t conv_act_size_c = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_w = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t in_h_start = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_w_start = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_start_in_h_curr_image = get_arg_val<uint32_t>(i); i+=1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w = get_compile_time_arg_val(3);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(4) - 1;
    // 5,6 not used
    constexpr uint32_t extra_padding_for_32B_alignment = get_compile_time_arg_val(7);
    //constexpr uint32_t act_block_width_padding_bytes = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const DataFormat data_format = get_dataformat(cb_id_act);
    uint32_t channel_stick_size = conv_act_size_c;
    uint32_t channel_stick_size_bytes = channel_stick_size << 1;

    const InterleavedPow2AddrGenFast<act_in_dram> s_act = {
        .bank_base_address = act_addr_dram_base,
        //.log_base_2_of_page_size = 5 // TODO: send as a compile-time arg, currently C=16 in FP16_B (so 32 B)
        //.log_base_2_of_page_size = 13 // TODO: send as a compile-time arg, currently C=16 x W=256 in FP16_B = 8192
        .log_base_2_of_page_size = 11 // TODO: send as a compile-time arg, currently C=4 x W=256 in FP16_B = 2048
    };
    uint32_t read_size_bytes = channel_stick_size_bytes << 3; // channel stick size * 8
    // Assumptions. Must be true. Validate on host.
    // assert(act_block_w_datums == C * weight_size_w)
    // assert(num_blocks_act_w == weight_size_h)
    // assert(act_block_w_datums % C == 0)
    // assert(act_block_w_datums % 32 == 0)
    // assert(act_block_h_datums % 32 == 0)
    // assert(act_block_h_ntiles == act_block_h_datums/32)
    // assert(act_block_w_ntiles == act_block_w_datums/32)
    // assert(act_block_num_tiles == (act_block_h_datums * act_block_w_datums)/1024)


    constexpr uint32_t in_w_padded_for_32_alignment = 231 + extra_padding_for_32B_alignment;

    uint32_t in_h = in_h_start;
    uint32_t in_h_reset = in_h;
    uint32_t out_w = out_w_start;
    uint32_t out_w_reset = out_w;
    uint32_t page_offset_h_2d_matrix = out_w_start * (channel_stick_size_bytes << 1);
    uint32_t page_offset_h_2d_matrix_reset = page_offset_h_2d_matrix;
    uint32_t last_start_in_h_stride = 222;
    uint32_t last_start_in_h_curr_image_reset = last_start_in_h_curr_image;
    for(uint32_t nbh = 0; nbh < num_blocks_act_h; nbh++) {
        uint32_t c_id_offset_inter_block_col = 0;
        uint32_t page_id_offset_inter_block_w = 0;
        for (uint32_t nbw = 0; nbw < num_blocks_act_w; nbw++) {
            out_w = out_w_reset;
            in_h = in_h_reset;
            page_offset_h_2d_matrix = page_offset_h_2d_matrix_reset;
            last_start_in_h_curr_image = last_start_in_h_curr_image_reset;
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            uint32_t l1_addr_offset = 0;
            for(uint32_t bh = 0; bh < act_block_h_datums; bh++) {
                //uint32_t c_id_offset_inra_block_col = 0;

                // channel_stick * filter window width is contiguous in page
                uint32_t page_id = in_h + page_id_offset_inter_block_w;
                uint32_t page_offset = page_offset_h_2d_matrix;
                uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                s_act.noc_async_read_partial_page(page_id, dst_addr, read_size_bytes, page_offset);
                l1_addr_offset += read_size_bytes;
                if(out_w < conv_output_size_w - 1) {
                    out_w += 1;
                    //first_c_id_in_2d_row += 2; // channel id stride in the w dimension
                    page_offset_h_2d_matrix += (channel_stick_size_bytes << 1); // * 2 for conv stride in the w dimension
                } else {
                    out_w = 0;
                    page_offset_h_2d_matrix = 0;
                    if (in_h < last_start_in_h_curr_image) {
                        in_h += 2; // stride_h
                    } else {
                        // next image in batch
                        // stride in_h for next image.. assume shape is 1, N*H, W, C, in_h represents h coordinate in this shape.
                        in_h += 8;
                        last_start_in_h_curr_image = in_h + last_start_in_h_stride;
                    }
                }
            } // for block height
            c_id_offset_inter_block_col += in_w_padded_for_32_alignment;
            page_id_offset_inter_block_w += 1;
            noc_async_read_barrier();
            cb_push_back(cb_id_act, act_block_num_tiles);
        } // for num of act blocks in inner width dim
        out_w_reset = out_w;
        in_h_reset = in_h;
        page_offset_h_2d_matrix_reset = page_offset_h_2d_matrix;
        last_start_in_h_curr_image_reset = last_start_in_h_curr_image;
    } // for num of act blocks in height dim
}
