// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
#include "debug_print.h"
inline void kernel_sleep(uint32_t loop_count = 1000) {
    for (volatile uint32_t i = 0; i < loop_count; ++ i);
}

inline void print_cb_details(uint32_t cb_id) {
    UNPACK((
    DPRINT << "cb_id " << cb_id << ": { "
            << "size: " << cb_interface[cb_id].fifo_size << ", "
            << "limit: " << cb_interface[cb_id].fifo_limit << ", "
            << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
            << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
            << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
            << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
            << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL() ));
}

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (int32_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << (uint) r << " :: " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}


namespace NAMESPACE {
void MAIN {


    uint32_t i = 0;

    uint32_t local_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_input_offset_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_prev_enabled = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_prev_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_next_enabled = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_next_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t input_cb_index = get_compile_time_arg_val(0); // sharded tiled
    uint32_t halo_prev_input_cb_index = get_compile_time_arg_val(1); // halo from prev core data tiled
    uint32_t halo_next_input_cb_index = get_compile_time_arg_val(2); // halo from next core data tiled
    uint32_t untilize_cb_index = get_compile_time_arg_val(3); // 1 row of tiles (untilized)
    uint32_t untilize_downsampled_cb_index = get_compile_time_arg_val(4); // full output size
    uint32_t tilize_out_cb_index = get_compile_time_arg_val(5); // final output = sharded output
    uint32_t num_input_tiles_in_row = get_compile_time_arg_val(6); // same for both local and halo inputs

    uint32_t num_output_rows_of_tiles = get_compile_time_arg_val(7);
    uint32_t num_output_tiles_in_row = get_compile_time_arg_val(8);
    uint32_t num_output_tiles = num_output_rows_of_tiles * num_output_tiles_in_row;

    untilize_init(input_cb_index, untilize_cb_index);
    pack_reconfig_data_format(untilize_cb_index);

    print_cb_details(input_cb_index);
    print_cb_details(untilize_cb_index);
    print_cb_details(untilize_downsampled_cb_index);
    print_cb_details(tilize_out_cb_index);

    if (halo_prev_enabled) {
        // not required since cb has same data format and untilize init already configured unpacker
        //untilize_init_short(halo_prev_input_cb_index);

        // Untilize halo input
        for(uint32_t b = 0; b < halo_prev_input_num_rows_of_tiles; ++ b) {

            cb_wait_front(halo_prev_input_cb_index, num_input_tiles_in_row);
            cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

            untilize_block(halo_prev_input_cb_index, num_input_tiles_in_row, untilize_cb_index);

            cb_push_back(untilize_cb_index, num_input_tiles_in_row);
            cb_pop_front(halo_prev_input_cb_index, num_input_tiles_in_row);
        }
    }

    // Unilize prev core's halo region - 1 row of tiles
    // untilize_block(prev_core_input_cb_index, num_input_tiles_per_block, untilize_cb_index);

    // Untilize input
    cb_pop_front(input_cb_index, local_input_offset_rows_of_tiles * num_input_tiles_in_row);
    for(uint32_t b = 0; b < local_input_num_rows_of_tiles; ++ b) {

        cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

        untilize_block(input_cb_index, num_input_tiles_in_row, untilize_cb_index);

        // print_full_tile(untilize_cb_index);

        cb_push_back(untilize_cb_index, num_input_tiles_in_row);
        cb_pop_front(input_cb_index, num_input_tiles_in_row);
    }

    if (halo_next_enabled) {
        // not required since cb has same data format and untilize init already configured unpacker
        //untilize_init_short(halo_next_input_cb_index);

        // Untilize halo input
        for(uint32_t b = 0; b < halo_next_input_num_rows_of_tiles; ++ b) {

            cb_wait_front(halo_next_input_cb_index, num_input_tiles_in_row);
            cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

            untilize_block(halo_next_input_cb_index, num_input_tiles_in_row, untilize_cb_index);

            cb_push_back(untilize_cb_index, num_input_tiles_in_row);
            cb_pop_front(halo_next_input_cb_index, num_input_tiles_in_row);
        }
    }
    untilize_uninit(input_cb_index);

    // // Tilize downsampled input
    // cb_wait_front(untilize_downsampled_cb_index, num_output_tiles);
    // cb_reserve_back(tilize_out_cb_index, num_output_tiles);

    tilize_init_short(untilize_downsampled_cb_index, num_output_tiles_in_row);
    pack_reconfig_data_format(tilize_out_cb_index);
    unpack_reconfig_data_format_srca(input_cb_index, untilize_downsampled_cb_index);

    for(uint32_t b=0;b<num_output_rows_of_tiles;++b)
    {
        cb_wait_front(untilize_downsampled_cb_index, num_output_tiles_in_row);
        cb_reserve_back(tilize_out_cb_index, num_output_tiles_in_row);

        // print_full_tile(untilize_downsampled_cb_index);

        tilize_block(untilize_downsampled_cb_index, num_output_tiles_in_row, tilize_out_cb_index);

        cb_push_back(tilize_out_cb_index, num_output_tiles_in_row);
        cb_pop_front(untilize_downsampled_cb_index, num_output_tiles_in_row);
    }
}
}
