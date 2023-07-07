#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "debug_print.h"

inline void noc_async_read_from_dram_to_l1(uint32_t dram_addr, uint32_t dram_noc_x, uint32_t dram_noc_y, uint32_t l1_dest_addr, uint32_t read_size) {
    uint64_t src_noc_addr = dataflow::get_noc_addr(dram_noc_x, dram_noc_y, dram_addr);
    dataflow::noc_async_read(src_noc_addr, l1_dest_addr, read_size);
}

inline void pad_l1_buffer_with_zeroes(uint32_t l1_addr, uint32_t pad_size_bytes) {
    volatile std::uint8_t* start_dst= (volatile uint8_t*)(l1_addr);
    for (uint32_t offset = 0; offset < pad_size_bytes; offset++) {
        *(start_dst + offset) = 0;
    }
}

void kernel_main() {
    uint32_t i = 0;
    uint32_t act_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_y = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_dram_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_dram_noc_y = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_c = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_weight_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_groups = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_matrix_height_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t weight_matrix_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_matrix_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_matrix_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_matrix_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_w_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_h_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_w_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t src_dram_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t dst_l1_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t src_dram_weight_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t dst_l1_weight_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t cb_id_weight = 1;
    uint32_t channel_stick_size = conv_act_size_c;
    uint32_t num_blocks_weight_h = num_blocks_act_w;
    uint32_t single_tile_size_bytes = 2048;
    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {

        // Read activations for this group
        // Activations are in channels last layout in dram
        {
            dataflow::cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t block_idx_h = (uint32_t) (group_idx / num_blocks_act_w) / (num_blocks_weight_w);
            uint32_t block_idx_w = (uint32_t) (group_idx % num_blocks_act_w);
            uint32_t block_idx = (block_idx_h * num_blocks_act_w) + block_idx_w;
            uint32_t start_block_2d_index_h = block_idx_h * act_block_h_datums;
            uint32_t start_block_2d_index_w = block_idx_w * act_block_w_datums;
            uint32_t start_block_2d_index = (start_block_2d_index_h * act_block_w_datums * num_blocks_act_w) + start_block_2d_index_w;
            uint32_t l1_write_addr_act = dataflow::get_write_ptr(cb_id_act);
            if(start_block_2d_index_w >= act_matrix_width_unpadded) {
                DPRINT << "Problem" << ENDL();
            }
            for(uint32_t h_b = 0; h_b < act_block_h_datums; h_b++) {
                uint32_t h = start_block_2d_index_h + h_b;
                uint32_t dst_address_offset_l1 = (h_b * act_block_w_datums)<<1;
                if (h >= act_matrix_height_unpadded) {
                    // pad (block shape padding for height dim)
                    uint32_t pad_size_bytes = act_block_w_datums<<1;
                    if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                        DPRINT << "Problem" << ENDL();
                    }
                    uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                    pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                }
                else {
                    uint32_t w = start_block_2d_index_w;
                    uint32_t end_block_2d_index_w = start_block_2d_index_w + act_block_w_datums - 1;
                    if(end_block_2d_index_w >= act_matrix_width) {
                        DPRINT << "Problem" << ENDL();
                    }
                    while (w <= end_block_2d_index_w) {
                        uint32_t src_address_offset_dram = 0;
                        uint32_t read_size_bytes = 0;
                        uint32_t pad = 0;
                        if (w >= act_matrix_width_unpadded) {
                            // pad (block shape padding for width dim)
                            if(end_block_2d_index_w != act_matrix_width-1) {
                                DPRINT << "Problem" << ENDL();
                            }
                            uint32_t pad_size_bytes = (end_block_2d_index_w - w + 1)<<1;
                            if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                DPRINT << "Problem" << ENDL();
                            }
                            uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                            pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            read_size_bytes = pad_size_bytes;
                        }
                        else {
                            uint32_t channel_stick_offset = w % channel_stick_size;
                            uint32_t channel_stick_col_id = w / channel_stick_size;
                            uint32_t channel_stick_row_id = h;
                            if(channel_stick_offset % 16 != 0) { // DRAM read address must be aligned to 32 bytes
                                DPRINT << "Problem" << ENDL();
                            }
                            uint32_t channel_stick_row_id_x = channel_stick_row_id % conv_output_size_w;
                            uint32_t channel_stick_row_id_y = channel_stick_row_id / conv_output_size_w;
                            uint32_t act_tensor_start_x = channel_stick_row_id_x * stride_w;
                            uint32_t act_tensor_start_y = channel_stick_row_id_y * stride_h;
                            uint32_t act_tensor_padded_x = act_tensor_start_x + (channel_stick_col_id % weight_size_w);
                            uint32_t act_tensor_padded_y = act_tensor_start_y + (channel_stick_col_id / weight_size_w);
                            if(w > end_block_2d_index_w) {
                                DPRINT << "Problem" << ENDL();
                            }
                            uint32_t a = channel_stick_size - channel_stick_offset;
                            uint32_t b = (end_block_2d_index_w+1)-w;
                            uint32_t read_size = a < b ? a : b;
                            read_size_bytes = read_size << 1;
                            if(act_tensor_padded_x < pad_w || act_tensor_padded_x >= (pad_w + conv_act_size_w) || act_tensor_padded_y < pad_h || act_tensor_padded_y >= (pad_h + conv_act_size_h)) {
                                // pad (conv padding)
                                uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                                uint32_t pad_size_bytes = read_size_bytes;
                                if(dst_address_offset_l1 + (pad_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                    DPRINT << "Problem" << ENDL();
                                }
                                pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            }
                            else {
                                uint32_t act_tensor_x = act_tensor_padded_x - pad_w;
                                uint32_t act_tensor_y = act_tensor_padded_y - pad_h;
                                if(act_tensor_x >= conv_act_size_w || act_tensor_y >= conv_act_size_h) {
                                    DPRINT << "Problem" << ENDL();
                                }
                                uint32_t act_tensor_channel_id = act_tensor_y * conv_act_size_w + act_tensor_x;
                                src_address_offset_dram = ((act_tensor_channel_id * channel_stick_size) + channel_stick_offset)<<1;
                                if(src_address_offset_dram % 32 != 0) { // DRAM read address must be aligned to 32 bytes
                                    DPRINT << "Problem1" << ENDL();
                                }
                                if(src_address_offset_dram >= src_dram_act_buffer_size_bytes) {
                                    DPRINT << "Problem2" << ENDL();
                                }
                                if(dst_address_offset_l1 + (read_size_bytes-1) >= dst_l1_act_buffer_size_bytes) {
                                    DPRINT << "Problem3" << ENDL();
                                }
                                uint32_t src_addr = act_addr_dram_base + src_address_offset_dram;
                                uint32_t dst_addr = l1_write_addr_act + dst_address_offset_l1;
                                noc_async_read_from_dram_to_l1(src_addr, act_dram_noc_x, act_dram_noc_y, dst_addr, read_size_bytes);
                            }
                        }
                        dst_address_offset_l1 += read_size_bytes;
                        w += (read_size_bytes>>1);
                        if(w > end_block_2d_index_w+1) {
                            DPRINT << "Problem" << ENDL();
                        }
                    }
                }
            }
        }

        // Read weights for this group
        // Weights are in tiled layout in dram
        {
            dataflow::cb_reserve_back(cb_id_weight, weight_block_num_tiles);
            uint32_t l1_write_addr_weight = dataflow::get_write_ptr(cb_id_weight);
            // Weight blocks are col major
            uint32_t block_idx_h = (uint32_t) (group_idx % num_blocks_weight_h);
            uint32_t block_idx_w = (uint32_t) (group_idx / num_blocks_weight_h) % (num_blocks_weight_w);
            uint32_t block_idx = (block_idx_w * num_blocks_weight_h) + block_idx_h;
            uint32_t start_block_tile_h_index = block_idx_h * weight_block_h_ntiles;
            uint32_t start_block_tile_w_index = block_idx_w * weight_block_w_ntiles;

            // Weight tiles are in row major order within block
            for(uint32_t tile_h_index_in_block = 0; tile_h_index_in_block < weight_block_h_ntiles; tile_h_index_in_block++) {
                for(uint32_t tile_w_index_in_block = 0; tile_w_index_in_block < weight_block_w_ntiles; tile_w_index_in_block++) {
                    uint32_t tile_index_h_in_matrix = tile_h_index_in_block + start_block_tile_h_index;
                    uint32_t tile_index_w_in_matrix = tile_w_index_in_block + start_block_tile_w_index;
                    // Weight tiles are in row major order in weight matrix in dram
                    uint32_t tile_index_in_matrix = (tile_index_h_in_matrix * weight_block_w_ntiles * num_blocks_weight_w) + tile_index_w_in_matrix;
                    if(tile_index_in_matrix >= weight_matrix_height_ntiles * weight_matrix_width_ntiles) {
                        DPRINT << "Problem" << ENDL();
                    }
                    // Weight tiles are in row major order in weight block in l1
                    uint32_t tile_index_in_block = tile_h_index_in_block * weight_block_w_ntiles + tile_w_index_in_block;
                    uint32_t src_address_offset_dram = tile_index_in_matrix * single_tile_size_bytes;
                    uint32_t read_size_bytes = single_tile_size_bytes;
                    uint32_t dst_address_offset_l1 = tile_index_in_block * single_tile_size_bytes;

                    if(src_address_offset_dram >= src_dram_weight_buffer_size_bytes) {
                        DPRINT << "Problem1" << ENDL();
                    }
                    if(dst_address_offset_l1 + (read_size_bytes-1) >= dst_l1_weight_buffer_size_bytes) {
                        DPRINT << "Problem2" << ENDL();
                    }
                    uint32_t src_addr = weight_addr_dram_base + src_address_offset_dram;
                    uint32_t dst_addr = l1_write_addr_weight + dst_address_offset_l1;
                    noc_async_read_from_dram_to_l1(src_addr, weight_dram_noc_x, weight_dram_noc_y, dst_addr, read_size_bytes);
                }
            }
        }
        dataflow::noc_async_read_barrier();
        dataflow::cb_push_back(cb_id_act, act_block_num_tiles);
        dataflow::cb_push_back(cb_id_weight, weight_block_num_tiles);
    }

}
