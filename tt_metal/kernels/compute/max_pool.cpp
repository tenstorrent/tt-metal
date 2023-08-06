#include <cstdint>

// #define DEBUG_PRINT 1

#include "compute_kernel_api.h"

// SliceRange srt = SliceRange{.h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 64, .ws = 8};
// SliceRange srr = SliceRange{.h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 64, .ws = 2};
// SliceRange sr = SliceRange{ .h0 = 31, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };

inline void tilize(uint32_t in_cb_id,
                   uint32_t in_ntiles_hw,
                   uint32_t in_ntiles_c,
                   uint32_t in_ntiles_hwc,
                   uint32_t window_hw_padded,
                   uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, 1);
    // PDPRINT("IN RM: " << in_ntiles_c * in_ntiles_hw << " :: " << TileSlice(in_cb_id, 0, srt, true, false));
    cb_reserve_back(out_cb_id, in_ntiles_hwc);
    tilize_init_short(in_cb_id, in_ntiles_hwc);
    tilize_block(in_cb_id, in_ntiles_hwc, out_cb_id);
    tilize_uninit();
    cb_push_back(out_cb_id, in_ntiles_hwc);
    cb_pop_front(in_cb_id, 1);
}

inline void reduce_h(uint32_t in_cb_id,
                     uint32_t in_scalar_cb_id,
                     uint32_t in_ntiles_hw,
                     uint32_t in_ntiles_c,
                     uint32_t in_ntiles_hwc,
                     uint32_t out_ntiles_c,
                     uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, in_ntiles_hwc);
    cb_reserve_back(out_cb_id, out_ntiles_c);
    reduce_init_delta_v2<false>(PoolType::MAX, ReduceDim::REDUCE_COL, out_cb_id);
    uint32_t base_tile_id = 0;
    for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i) {
        // add to accumulator all the in_ntiles_hw in a column of tiles
        acquire_dst(tt::DstMode::Half);
        for (uint32_t hw_i = 0; hw_i < in_ntiles_hw; ++hw_i) {
            uint32_t tile_i = base_tile_id + hw_i;
            reduce_tile_v2(PoolType::MAX, ReduceDim::REDUCE_COL, in_cb_id, in_scalar_cb_id, tile_i, 0, tile_i);
        }
        pack_tile(c_i, out_cb_id);
        release_dst(tt::DstMode::Half);
        base_tile_id += in_ntiles_hw;
    }
    reduce_revert_delta_v2(out_cb_id);
    cb_push_back(out_cb_id, out_ntiles_c);
    cb_pop_front(in_cb_id, in_ntiles_hwc);
}

inline void untilize(uint32_t in_cb_id,
                     uint32_t out_ntiles_c,
                     uint32_t out_npages,
                     uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, out_ntiles_c);
    cb_reserve_back(out_cb_id, out_npages);
    untilize_init_short(in_cb_id);
    untilize_block(in_cb_id, out_ntiles_c, out_cb_id);
    untilize_uninit(in_cb_id);
    cb_push_back(out_cb_id, out_npages);
    cb_pop_front(in_cb_id, out_ntiles_c);
}  // untilize()

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;
    constexpr uint32_t in_tiled_cb_id = tt::CB::c_intermed0;
    constexpr uint32_t out_tiled_cb_id = tt::CB::c_intermed1;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    const uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    const uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    const uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    const uint32_t window_hw_padded = get_compile_time_arg_val(3);
    const uint32_t out_h = get_compile_time_arg_val(4);
    const uint32_t out_w = get_compile_time_arg_val(5);
    const uint32_t out_ntiles_hw = get_compile_time_arg_val(6);
    const uint32_t out_ntiles_c = get_compile_time_arg_val(7);

    tilize_init(in_cb_id, in_ntiles_hwc, in_tiled_cb_id);
    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t out_h_i = 0; out_h_i < out_h; ++out_h_i) {
        for (uint32_t out_w_i = 0; out_w_i < out_w; ++out_w_i) {
            // NOTE: Assuming in_ntiles_hw < 8 for now.
            // TODO: subblocking to support this.
            // tilize
            tilize(in_cb_id, in_ntiles_hw, in_ntiles_c, in_ntiles_hwc, window_hw_padded, in_tiled_cb_id);
            // Reduce H
            reduce_h(in_tiled_cb_id, in_scalar_cb_id, in_ntiles_hw, in_ntiles_c, in_ntiles_hwc, out_ntiles_c, out_tiled_cb_id);
            // untilize
            untilize(out_tiled_cb_id, out_ntiles_c, 1, out_cb_id);
        }
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
