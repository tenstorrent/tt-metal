// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// #include "compute_kernel_api.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
// #include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug_macros.h"

SliceRange srt = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 4};
SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
SliceRange srr1 = SliceRange{.h0 = 1, .h1 = 2, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1};
SliceRange src = SliceRange{.h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PDPRINT("======");
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = r + 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PDPRINT((uint)r << TileSlice(cb_id, tile_id, sr, true, untilize));
    }
    PDPRINT("++++++");
}

inline void print_cb_details(uint32_t cb_id) {
    PDPRINT(
        "cb_id " << cb_id << ": { "
                 << "size: " << get_local_cb_interface(cb_id).fifo_size << ", "
                 << "limit: " << get_local_cb_interface(cb_id).fifo_limit << ", "
                 << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", "
                 << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", "
                 << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", "
                 << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << ", "
                 << "wr_tile_ptr: " << get_local_cb_interface(cb_id).fifo_wr_tile_ptr << " }");
}
#endif

inline void tilize(
    uint32_t out_nelems,
    uint32_t in_cb_id,
    uint32_t in_ntiles_hw,
    uint32_t in_ntiles_c,
    uint32_t in_ntiles_hwc,
    uint32_t window_hw_padded,
    uint32_t out_cb_id) {
    tilize_init_short(in_cb_id, in_ntiles_hwc, out_cb_id);
    for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++out_elem_i) {
        cb_wait_front(in_cb_id, 1);
        cb_reserve_back(out_cb_id, in_ntiles_hwc);
        tilize_block(
            in_cb_id,
            in_ntiles_hwc,
            out_cb_id);  // TODO: need to ensure the ordering for reduction when in_ntiles_hw > 1
        // print_full_tile(in_cb_id, 0, false);
        // PDPRINT("OUT TILE :: " << TileSlice(out_cb_id, 0, srr, true, true));
        // print_cb_details(in_cb_id);
        cb_push_back(out_cb_id, in_ntiles_hwc);
        cb_pop_front(in_cb_id, 1);
    }
    tilize_uninit(in_cb_id, out_cb_id);
}

inline void reduce_h(
    uint32_t out_nelems,
    uint32_t in_cb_id,
    uint32_t in_scalar_cb_id,
    uint32_t in_ntiles_hw,
    uint32_t in_ntiles_c,
    uint32_t in_ntiles_hwc,
    uint32_t out_ntiles_c,
    uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, in_ntiles_hwc * out_nelems);
    cb_reserve_back(out_cb_id, out_ntiles_c * out_nelems);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_COL>(out_cb_id);
    uint32_t base_tile_id = 0;
    for (uint32_t c_i = 0; c_i < in_ntiles_c * out_nelems; ++c_i) {
        // add to accumulator all the in_ntiles_hw in a column of tiles
        acquire_dst();
        uint32_t dst_i = 0;  // TODO [AS]: Use more than one dst tile at a time
        for (uint32_t hw_i = 0; hw_i < in_ntiles_hw; ++hw_i) {
            uint32_t tile_i = base_tile_id + hw_i;
            reduce_tile(in_cb_id, in_scalar_cb_id, tile_i, 0, dst_i);
        }
        pack_tile(dst_i, out_cb_id);
        release_dst();
        base_tile_id += in_ntiles_hw;
    }
    reduce_revert_delta(out_cb_id);
    cb_push_back(out_cb_id, out_ntiles_c * out_nelems);
    cb_pop_front(in_cb_id, in_ntiles_hwc * out_nelems);
}

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in_scalar_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t in_tiled_cb_id = tt::CBIndex::c_24;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;

    const uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    const uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    const uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    const uint32_t window_hw_padded = get_compile_time_arg_val(3);
    const uint32_t out_h = get_compile_time_arg_val(4);
    const uint32_t out_w = get_compile_time_arg_val(5);
    const uint32_t out_ntiles_c = get_compile_time_arg_val(7);
    const uint32_t out_nelems = get_compile_time_arg_val(8);
    const uint32_t out_w_loop_count = get_compile_time_arg_val(9);
    const uint32_t nbatch = get_compile_time_arg_val(10);
    const uint32_t out_h_per_core = get_compile_time_arg_val(11);

    tilize_init(in_cb_id, in_ntiles_hwc, in_tiled_cb_id);

#if DEBUG_PRINT == 1
    print_cb_details(in_cb_id);
    print_cb_details(in_scalar_cb_id);
    print_cb_details(in_tiled_cb_id);
    print_cb_details(out_cb_id);
#endif

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t batch = 0; batch < nbatch; ++batch) {
        for (uint32_t out_h_i = 0; out_h_i < out_h_per_core; ++out_h_i) {
            for (uint32_t out_w_i = 0; out_w_i < out_w_loop_count; ++out_w_i) {
                // NOTE: Assuming in_ntiles_hw < 8 for now.
                // TODO: subblocking to support this.
                // UDPRINT('T' << out_w_i);
                // tilize
                tilize(
                    out_nelems, in_cb_id, in_ntiles_hw, in_ntiles_c, in_ntiles_hwc, window_hw_padded, in_tiled_cb_id);
                // UDPRINT('R' << out_w_i);
                // Reduce H
                reduce_h(
                    out_nelems,
                    in_tiled_cb_id,
                    in_scalar_cb_id,
                    in_ntiles_hw,
                    in_ntiles_c,
                    in_ntiles_hwc,
                    out_ntiles_c,
                    out_cb_id);
            }
        }
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
