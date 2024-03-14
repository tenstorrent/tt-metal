// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// #include "compute_kernel_api.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
// #include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
    #include "debug/dprint.h"
    // #include "debug_macros.h"

    // SliceRange srt = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 4};
    // SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
    // SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
    // SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

    inline void print_tile_rows(uint32_t cb_id, uint32_t rows = 32, uint32_t tile_id = 0, bool untilize = false) {
        // UNPACK(( DPRINT << "======" << ENDL() ));
        for (uint16_t r = 0; r < rows; ++ r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            // UNPACK(( DPRINT << (uint)r << " :: " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
            UNPACK(( DPRINT << (uint)r << " :: " << TileSlice(cb_id, tile_id, sr, true, untilize) ));
        }
        // UNPACK(( DPRINT << "++++++" << ENDL() ));
    }

    inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
        UNPACK(( DPRINT << "======" << ENDL() ));
        for (uint16_t r = 0; r < 32; ++ r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            UNPACK(( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
        }
        UNPACK(( DPRINT << "++++++" << ENDL() ));
    }

    // inline void print_cb_details(uint32_t cb_id) {
    //     DPRINT << "cb_id " << cb_id << ": { "
    //             << "size: " << cb_interface[cb_id].fifo_size << ", "
    //             << "limit: " << cb_interface[cb_id].fifo_limit << ", "
    //             << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
    //             << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
    //             << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
    //             << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
    //             << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
    // }
#endif

template<uint32_t in_ntiles_hw, uint32_t in_ntiles_c, uint32_t out_ntiles_c, uint32_t nblocks, bool is_partial_tile>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_ntiles_hwc,
    const uint32_t out_cb_id) {

    constexpr uint32_t effective_nblocks = is_partial_tile ? 1 : nblocks;
    constexpr uint32_t num_output_tiles = out_ntiles_c * effective_nblocks;
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 2 /*fixme 1*/ : 2;
    constexpr uint32_t num_out_rows = 1;
    cb_reserve_back(out_cb_id, out_ntiles_c * effective_nblocks);
    tile_regs_acquire();
    for (uint32_t out_elem_i = 0; out_elem_i < effective_nblocks; ++ out_elem_i) {
        cb_wait_front(in_cb_id, 1);
        unpack_tilizeA_B_block(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, 0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/, num_faces_in_tile /* unpack 1 or 2 faces ) */);
        for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i) {
            reduce_tile_math(in_ntiles_c * out_elem_i + c_i,  num_faces_in_tile /* reduce 1 or 2 faces */);
        }
        cb_pop_front(in_cb_id, 1);
    }

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(out_cb_id, 1/*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile);  /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();

    cb_push_back(out_cb_id, out_ntiles_c * effective_nblocks);
}

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;
    constexpr uint32_t in_tiled_cb_id = tt::CB::c_intermed0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    const uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    const uint32_t window_hw_padded = get_compile_time_arg_val(3);
    const uint32_t out_h = get_compile_time_arg_val(4);
    const uint32_t out_w = get_compile_time_arg_val(5);
    const uint32_t out_ntiles_c = get_compile_time_arg_val(7);
    const uint32_t nblocks = get_compile_time_arg_val(8);
    const uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(13);
    const uint32_t in_c = get_compile_time_arg_val(14);
    const uint32_t num_output_tiles = out_ntiles_c * nblocks;

    // const uint32_t TILE_WIDTH = 32;
    const bool is_partial_tile = in_c < 32;
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 2 /*fixme 1*/ : 2;
    constexpr uint32_t num_out_rows = 1;

    tilizeA_B_reduce_init(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_cb_id, num_faces_in_tile);
    pack_untilize_dst_init_short<num_output_tiles>(out_cb_id, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core_by_nblocks; ++ i) {
        // NOTE: Assuming in_ntiles_hw < 8 for now.
        // TODO: subblocking to support this.
        reduce_h_fused<in_ntiles_hw, in_ntiles_c, out_ntiles_c, nblocks, is_partial_tile>(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_cb_id);
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
