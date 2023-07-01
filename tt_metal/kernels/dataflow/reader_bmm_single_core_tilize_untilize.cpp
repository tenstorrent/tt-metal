#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"
#include "debug_print_tile.h"

/**
 * Reader kernel used for single core BMM with tilize activations and untilize output.
 */
void kernel_main() {
    // in0
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_block_h = get_arg_val<uint32_t>(1);
    uint32_t in0_num_blocks_h = get_arg_val<uint32_t>(2);
    uint32_t in0_num_blocks_w = get_arg_val<uint32_t>(3);           // == in1_num_blocks_h
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(4);
    uint32_t in0_block_nrows = get_arg_val<uint32_t>(5);
    uint32_t in0_start_row_id = get_arg_val<uint32_t>(6);
    uint32_t in0_row_size_bytes = get_arg_val<uint32_t>(7);         // size of a full row
    uint32_t in0_read_row_size_bytes = get_arg_val<uint32_t>(8);    // size of partial row to fit within a block width

    // in1
    uint32_t in1_addr = get_arg_val<uint32_t>(9);
    uint32_t in1_block_h = get_arg_val<uint32_t>(10);
    uint32_t in1_block_w = get_arg_val<uint32_t>(11);
    uint32_t in1_num_blocks_w = get_arg_val<uint32_t>(12);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(13);
    uint32_t in1_stride_h = get_arg_val<uint32_t>(14);
    uint32_t in1_next_block_stride_h = get_arg_val<uint32_t>(15);
    uint32_t in1_next_block_stride_w = get_arg_val<uint32_t>(16);

    uint32_t in0_block_w = in1_block_h;

    // DPRINT << "in0_addr (in0_dram_addr): " << in0_addr << ENDL();
    // DPRINT << "in0_block_h (num_tiles): " << in0_block_h << ENDL();
    // DPRINT << "in0_num_blocks_h: " << in0_num_blocks_h << ENDL();
    // DPRINT << "in0_num_blocks_w: " << in0_num_blocks_w << ENDL();
    // DPRINT << "in0_block_num_tiles: " << in0_block_num_tiles << ENDL();
    // DPRINT << "in0_block_nrows: " << in0_block_nrows << ENDL();
    // DPRINT << "in0_start_row_id: " << in0_start_row_id << ENDL();
    // DPRINT << "in0_row_size_bytes (in0_width * dtype_nbyte): " << in0_row_size_bytes << ENDL();
    // DPRINT << "in0_read_row_size_bytes (in0_block_w * TILE_WIDTH * dtype_nbytes): " << in0_read_row_size_bytes << ENDL();
    // DPRINT << "in1_addr (in1_dram_addr): " << in1_addr << ENDL();
    // DPRINT << "in1_block_h (num_tiles): " << in1_block_h << ENDL();
    // DPRINT << "in1_block_w (num_tiles): " << in1_block_w << ENDL();
    // DPRINT << "in1_num_blocks_w: " << in1_num_blocks_w << ENDL();
    // DPRINT << "in1_block_num_tiles: " << in1_block_num_tiles << ENDL();
    // DPRINT << "in1_stride_h (in1_width_ntiles = in1_num_blocks_w * in1_block_w): " << in1_stride_h << ENDL();
    // DPRINT << "in1_next_block_stride_h: (in1_width_ntiles * in1_block_h)" << in1_next_block_stride_h << ENDL();
    // DPRINT << "in1_next_block_stride_w (in1_block_w): " << in1_next_block_stride_w << ENDL();


    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;
    constexpr uint32_t in2_cb_id = tt::CB::c_in2;

    constexpr uint32_t dtype_nbytes = 2;
    constexpr uint32_t TILE_HEIGHT = 32;                            // TODO: use a common source of truth
    constexpr uint32_t TILE_WIDTH = 32;                             // TODO: use a common source of truth
    const uint32_t tile_size_bytes = get_tile_size(in0_cb_id);      // == get_tile_size(in1_cb_id)

    DPRINT << "tile_size_bytes: " << tile_size_bytes << ENDL();

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = in0_addr,
        .page_size = in0_row_size_bytes
    };

    constexpr uint32_t tile_size_pow2_exponent = 11;    // 2^11 = 2048 = 32 * 32 * 2 bytes, tile size for 2 byte data types
    const InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = in1_addr,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    bool one_time_rowmajor_start = true;
    bool one_time_rowmajor_end = true;
    bool one_time_tilemajor_start = true;
    bool one_time_tilemajor_end = true;

    uint32_t dim_x = 512 * 1 * 1;   // Each noc_async_read tile size
    uint32_t dim_y = 256*256*2/dim_x;

    const InterleavedPow2AddrGen<true> s2 = {
        .bank_base_address = in0_addr,
        .log_base_2_of_page_size = 8
    };

    const InterleavedAddrGen<true> s3 = {
        .bank_base_address = in0_addr,
        .page_size = dim_x
    };

    // kernel_profiler::mark_time(5);
    uint32_t in0_curr_block_start_row_id = in0_start_row_id;
    // loop over in0 blocks along h
    for(uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        // Reset in1 (weight) start tile index
        uint32_t in1_start_tile_id = 0;
        // loop over in1 blocks along w
        for(uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
            uint32_t in1_current_block_start_tile_id = in1_start_tile_id;
            // reset offset
            uint32_t in0_row_offset_bytes = 0;
            // loop over in0 blocks along w (== in1 blocks along h)
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                // kernel_profiler::mark_time(6);
                // read in input data for current block

                // in0 DRAM -> L1 (activations in row major form)
                // partial rows are read since multiple blocks can span along the rows
                cb_reserve_back(in0_cb_id, in0_block_num_tiles);
                // kernel_profiler::mark_time(7);
                uint32_t in0_write_l1_addr = get_write_ptr(in0_cb_id);
                uint32_t in0_curr_block_row_id = in0_curr_block_start_row_id;
                // loop over in0 block tiles along h

                kernel_profiler::mark_time(5);
                for (uint32_t in0_curr_row_bank_id = 0; in0_curr_row_bank_id < dim_y; ++in0_curr_row_bank_id) {
                    uint64_t in0_row_noc_addr = get_noc_addr(in0_curr_row_bank_id, s3, in0_row_offset_bytes);
                    DPRINT << "in0_row_noc_addr: " << in0_row_noc_addr << " in0_write_l1_addr: " << in0_write_l1_addr << ENDL();
                    noc_async_read(in0_row_noc_addr, in0_write_l1_addr, dim_x);
                    in0_write_l1_addr += dim_x;
                }
                kernel_profiler::mark_time(6);



//                 kernel_profiler::mark_time(5);
// DPRINT << "\nStart NIU Programming\n";

// DPRINT << "Finish NIU Programming\n";
//                 kernel_profiler::mark_time(6);



                // for (uint32_t in0_tile_h_i = 0; in0_tile_h_i < in0_block_h; ++in0_tile_h_i) {
                //     uint32_t in0_curr_row_bank_id = in0_curr_block_row_id;
                //     // loop over each row of the tile
                //     for (uint32_t in0_row_h_i = 0; in0_row_h_i < TILE_HEIGHT; ++in0_row_h_i) {
                // // kernel_profiler::mark_time_once(11, &one_time_rowmajor_start);
                //         uint64_t in0_row_noc_addr = get_noc_addr(in0_curr_row_bank_id, s0, in0_row_offset_bytes);
                //         noc_async_read(in0_row_noc_addr, in0_write_l1_addr, in0_read_row_size_bytes);
                //         in0_write_l1_addr += in0_read_row_size_bytes;
                //         ++in0_curr_row_bank_id;
                // // kernel_profiler::mark_time_once(12, &one_time_rowmajor_end);
                //     }
                //     in0_curr_block_row_id += TILE_HEIGHT;
                // } // for in0_block_h


                // kernel_profiler::mark_time(8);
                noc_async_read_barrier();

DPRINT << "Pass Barrier\n";

                kernel_profiler::mark_time(7);

                // for (int k = 0; k < 32; k++) {
                //     auto s8 = SliceRange{ .h0 = k, .h1 = k+1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 };
                //     DPRINT << TSLICE(in0_cb_id, 0, s8);
                // }


                // DPRINT << "IN0 BLOCK: " << in0_block_w_i << "," << in0_block_h_i << ENDL();
                // auto slice0 = SliceRange{ .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 };
                // DPRINT  << TSLICE(in0_cb_id, 0, slice0) << ENDL();

                in0_row_offset_bytes += in0_read_row_size_bytes;
                cb_push_back(in0_cb_id, in0_block_num_tiles);

                // in1 DRAM -> L1 (weights in tiled form)
                cb_reserve_back(in1_cb_id, in1_block_num_tiles);
                // kernel_profiler::mark_time(9);
                uint32_t in1_write_l1_addr = get_write_ptr(in1_cb_id);
                uint32_t in1_row_start_tile_id = in1_current_block_start_tile_id;
                // loop over in1 block tiles along h
                for(uint32_t in1_tile_h_i = 0; in1_tile_h_i < in1_block_h; ++in1_tile_h_i) {
                    uint32_t in1_tile_id = in1_row_start_tile_id;
                    // loop over in1 block tiles along w
                    for(uint32_t in1_tile_w_i = 0; in1_tile_w_i < in1_block_w; ++in1_tile_w_i) {
                // kernel_profiler::mark_time_once(13, &one_time_tilemajor_start);
                        uint64_t in1_tile_noc_addr = get_noc_addr(in1_tile_id, s1);
                        noc_async_read(in1_tile_noc_addr, in1_write_l1_addr, tile_size_bytes);
                        in1_write_l1_addr += tile_size_bytes;
                        in1_tile_id += 1;
                // kernel_profiler::mark_time_once(14, &one_time_tilemajor_end);
                    } // for in1_block_w
                    in1_row_start_tile_id += in1_stride_h;
                } // for in1_block_h
                // kernel_profiler::mark_time(10);
                noc_async_read_barrier();

                // DPRINT << "IN1 BLOCK: " << in1_block_w_i << "," << in0_block_w_i << ENDL();
                // auto slice1 = SliceRange{ .h0 = 0, .h1 = 32, .hs = 16, .w0 = 0, .w1 = 32, .ws = 16 };
                // DPRINT  << TSLICE(in1_cb_id, 0, slice1) << ENDL();

                in1_current_block_start_tile_id += in1_next_block_stride_h;
                cb_push_back(in1_cb_id, in1_block_num_tiles);
            } // for in0_num_blocks_w
            in1_start_tile_id += in1_next_block_stride_w;
        } // for in1_num_blocks_w
        in0_curr_block_start_row_id += in0_block_nrows;
    } // for in0_num_blocks_h
} // kernel_main()
