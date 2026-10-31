// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC of the deduplicated MoE stream with gate|up and down in one launch (tt/moe_stream.py FusedStream).
//
// Gate|up as moe_gate_up_uniq_compute.cpp: per distinct expert, [rows, 32] x tiles @ [K, gate | up] (ct = 2, LoFi,
// fp32 dest) -> silu(gate) * up -> times its routing-weight tile -> one [rows, 32] bf16 h tile (the writer sends it to
// every bank core).
//
// Down: once every core's h of every expert is in cb_hall (expert j's h_kt tiles at j * h_kt), the sum over experts of
// h_j @ w_down_j[:, this core's cols output tiles], all cols tiles accumulating in fp32 dest at once (the Compact
// layout: 16 rows per [rows <= 8, 32] tile), the down blocks arriving expert by expert, down_rows K rows each.
//
// Two row halves (16 users): a listing of half 1 (the listings past half 0's count) multiplies x rows 8-15 (cb_x's
// second kt tiles); down runs twice, over half 0's listings and then half 1's, each packing its own cols tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "moe_stream_common.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_silu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

using namespace ckernel;
#ifdef TRISC_MATH
using namespace ckernel::sfpu;
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(4);
    constexpr uint32_t blocks_per_pair = get_compile_time_arg_val(5);
    constexpr uint32_t block_k = get_compile_time_arg_val(6);
    constexpr uint32_t rows = get_compile_time_arg_val(7);
    constexpr uint32_t cb_hall = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t h_kt = get_compile_time_arg_val(10);
    constexpr uint32_t down_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t down_rows = get_compile_time_arg_val(12);  // K rows per down block
    constexpr uint32_t cols = get_compile_time_arg_val(13);
    // Timing experiments only (MIMO_MOE_STREAM_DEBUG; wrong results): 2 = no matmuls, the weight blocks are only
    // waited for and popped (the stream's own rate).
    constexpr uint32_t debug = get_compile_time_arg_val(14);
    constexpr uint32_t halves = get_compile_time_arg_val(15);
    constexpr bool mm = debug != 2;
    constexpr uint32_t ct = 2;  // gate, up
    constexpr uint32_t block_tiles = block_k * ct;
    constexpr uint32_t kt = block_k * blocks_per_pair;
    constexpr uint32_t silu_iterations = rows / 2;  // SFPU rows per face iteration: 2
    static_assert(down_rows * cols == block_tiles && down_blocks * down_rows == h_kt);

    MOE_MARK(30);
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<false, dest_layout>(cb_x, cb_w, cb_h, ct);
    cb_wait_front(cb_x, halves * kt);
    cb_wait_front(cb_cnt, 1);
    const uint32_t n = read_tile_value(cb_cnt, 0, 0);
    const uint32_t n0 = halves == 1 ? n : read_tile_value(cb_cnt, 0, 1);  // half 0's listings
    cb_pop_front(cb_cnt, 1);
    MOE_MARK(31);

    for (uint32_t j = 0; j < n; j++) {
        const uint32_t x0 = j < n0 ? 0 : kt;  // the listing's half's x tiles
        tile_regs_acquire();
        for (uint32_t b = 0; b < blocks_per_pair - 1; b++) {
            cb_wait_front(cb_w, block_tiles);
            if constexpr (mm) {
                custom_mm_block<dest_layout, false>(cb_x, cb_w, x0 + b * block_k, 0, 0, block_k, ct);
            }
            cb_pop_front(cb_w, block_tiles);
        }
        cb_wait_front(cb_w, block_tiles);
        if constexpr (mm) {
            custom_mm_block<dest_layout>(cb_x, cb_w, x0 + (blocks_per_pair - 1) * block_k, 0, 0, block_k, ct);
        }
        cb_pop_front(cb_w, block_tiles);

        // Only rows 0..rows-1 are packed (tiny [rows, 32] h tile): silu runs on those rows of faces 0 and 1 only;
        // the cheap multiplies cover the whole dest tile.
        silu_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, silu_iterations), 0, VectorMode::R));
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        // The routing weights: the expert's weight tile into dest 1 (the up tile is consumed), then a row-wise
        // multiply.
        cb_wait_front(cb_wt, 1);
        reconfig_data_format_srca(cb_w, cb_wt);
        copy_tile_to_dst_init_short(cb_wt);
        copy_tile(cb_wt, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_wt, 1);
        reconfig_data_format_srca(cb_wt, cb_w);

        cb_reserve_back(cb_h, 1);
        tile_regs_wait();
        pack_tile(0, cb_h);
        tile_regs_release();
        cb_push_back(cb_h, 1);

        // The SFPU and copy inits reprogram address modes and the unpacker the matmul relies on.
        custom_mm_block_init_short<false, dest_layout>(cb_x, cb_w, cb_h, ct);
        if (j == 0) {
            MOE_MARK(32);
        }
    }
    cb_pop_front(cb_x, halves * kt);
    MOE_MARK(33);

    // Down: every expert's blocks accumulate into the same cols dest tiles.
    constexpr auto down_layout = CustomMmDestLayout::Compact;
    custom_mm_block_uninit<dest_layout>(cb_h);
    custom_mm_block_init_short<false, down_layout>(cb_hall, cb_w, cb_out, cols);
    cb_wait_front(cb_hall, n * h_kt);
    MOE_MARK(34);
    for (uint32_t h = 0; h < halves; h++) {
        const uint32_t lo = h == 0 ? 0 : n0;
        const uint32_t hi = h == 0 ? n0 : n;
        tile_regs_acquire();
        for (uint32_t j = lo; j < hi; j++) {
            for (uint32_t b = 0; b < down_blocks; b++) {
                cb_wait_front(cb_w, block_tiles);
                if constexpr (mm) {
                    custom_mm_block<down_layout>(cb_hall, cb_w, j * h_kt + b * down_rows, 0, 0, down_rows, cols);
                }
                cb_pop_front(cb_w, block_tiles);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, cols);
        tile_regs_wait();
        for (uint32_t c = 0; c < cols; c++) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, cols);
    }
    custom_mm_block_uninit<down_layout>(cb_out);
    cb_pop_front(cb_hall, n * h_kt);
    MOE_MARK(35);
}
