// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

namespace ckernel {

// When fp32_dest_acc_en is set, unpack/math must be explicitly reconfigured between ops (see FP32_DEST_ACC_EN).
ALWI void mul_tiles_init_with_dt(uint32_t icb0, uint32_t icb1) {
    reconfig_data_format(icb0, icb1);
    mul_init(icb0, icb1);
}

ALWI void sub_bcast_cols_init_with_dt(uint32_t icb0, uint32_t icb1) {
    reconfig_data_format(icb0, icb1);
    sub_bcast_cols_init(icb0, icb1);
}

}  // namespace ckernel

constexpr uint32_t DST_REG_ID = 0;
constexpr uint32_t ONE_TILE = 1;

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)

// Accumulate y * grad for a complete tile-row in FP32 DEST, then reduce its 32
// columns in-place. The resulting per-row scalar remains in column 0 and is
// packed once to the Float32 sum CB. This avoids the old partial-CB FPU reload.
template <bool RetainInputs>
ALWI void mul_reduce_row_to_scalar(
    uint32_t y_cb_id, uint32_t grad_cb_id, uint32_t sum_cb_id, uint32_t num_tiles_per_row, uint32_t tiles_per_block) {
    constexpr uint32_t accum_reg = 0;
    constexpr uint32_t y_reg = 1;
    constexpr uint32_t grad_reg = 2;
    constexpr uint32_t product_reg = 3;

    reconfig_data_format_srca(y_cb_id);
    tile_regs_acquire();
    bool first_tile = true;
    for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
        cb_wait_front(y_cb_id, tiles_per_block);
        cb_wait_front(grad_cb_id, tiles_per_block);
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            copy_init(y_cb_id);
            copy_tile(y_cb_id, i, y_reg);
            copy_init(grad_cb_id);
            copy_tile(grad_cb_id, i, grad_reg);

            mul_binary_tile_init();
            if (first_tile) {
                mul_binary_tile(y_reg, grad_reg, accum_reg);
                first_tile = false;
            } else {
                mul_binary_tile(y_reg, grad_reg, product_reg);
                add_binary_tile_init();
                add_binary_tile(accum_reg, product_reg, accum_reg);
            }
        }
        if constexpr (!RetainInputs) {
            cb_pop_front(y_cb_id, tiles_per_block);
            cb_pop_front(grad_cb_id, tiles_per_block);
        }
    }

    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(accum_reg);
    tile_regs_commit();
    pack_and_push(accum_reg, sum_cb_id);
}

ALWI void compute_output_tile(
    uint32_t y_cb_id, uint32_t grad_cb_id, uint32_t sum_cb_id, uint32_t out_cb_id, uint32_t tile_idx) {
    constexpr uint32_t sum_reg = 0;
    constexpr uint32_t grad_reg = 1;
    constexpr uint32_t y_reg = 2;

    reconfig_data_format_srca(sum_cb_id);
    tile_regs_acquire();
    copy_init(sum_cb_id);
    copy_tile(sum_cb_id, 0, sum_reg);
    reconfig_data_format_srca(grad_cb_id);
    copy_init(grad_cb_id);
    copy_tile(grad_cb_id, tile_idx, grad_reg);
    copy_init(y_cb_id);
    copy_tile(y_cb_id, tile_idx, y_reg);

    sfpu_sub_bcast_col(grad_reg, sum_reg);
    mul_binary_tile_init();
    mul_binary_tile(y_reg, grad_reg, grad_reg);

    tile_regs_commit();
    pack_and_push(grad_reg, out_cb_id);
}

#else

// Stream y * grad through the row, mul-accumulating elementwise into DST[0].
// `mul_init` programs ELWMUL with acc_to_dest=true, so within a single
// tile_regs_acquire/commit window each `mul_tiles(y, grad, i, i, 0)` performs
//   DST[0] += y[i] * grad[i]
// (FP32 in DST when fp32_dest_acc_en). After all tiles, DST[0] holds 32 column
// partials per row: column j = sum over k of input[k*32 + j]. The caller collapses
// those 32 partials into a per-row scalar with a single matmul-with-ones.
//
// `RetainInputs` mirrors the reader contract: when the row fits in L1 the inputs
// are kept for pass 2, otherwise they are popped block-by-block as we consume them.
template <bool RetainInputs>
ALWI void mul_accumulate_row_to_dst(
    uint32_t y_cb_id,
    uint32_t grad_cb_id,
    uint32_t partial_cb_id,
    uint32_t num_tiles_per_row,
    uint32_t tiles_per_block) {
    ckernel::mul_tiles_init_with_dt(y_cb_id, grad_cb_id);

    tile_regs_acquire();
    for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
        cb_wait_front(y_cb_id, tiles_per_block);
        cb_wait_front(grad_cb_id, tiles_per_block);
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            mul_tiles(y_cb_id, grad_cb_id, i, i, DST_REG_ID);
        }
        if constexpr (!RetainInputs) {
            cb_pop_front(y_cb_id, tiles_per_block);
            cb_pop_front(grad_cb_id, tiles_per_block);
        }
    }
    tile_regs_commit();
    pack_and_push(DST_REG_ID, partial_cb_id);
}

// Collapse the 32 column partials per row in `partial_cb_id` into a single per-row
// scalar in `sum_cb_id` via one matmul with the ones tile.
ALWI void reduce_partial_to_scalar(uint32_t partial_cb_id, uint32_t ones_cb_id, uint32_t sum_cb_id) {
    tile_regs_acquire();
    ckernel::reconfig_data_format(ones_cb_id, partial_cb_id);
    matmul_init(partial_cb_id, ones_cb_id, /*transpose*/ 0);

    cb_wait_front(partial_cb_id, ONE_TILE);
    matmul_tiles(partial_cb_id, ones_cb_id, 0, 0, DST_REG_ID);
    tile_regs_commit();

    cb_pop_front(partial_cb_id, ONE_TILE);
    pack_and_push(DST_REG_ID, sum_cb_id);
}

// Fused subtract and multiply: output = y * (grad - sum)
// Reuses DST register to eliminate intermediate CB write/read
ALWI void fused_sub_mul(
    uint32_t y_cb_id,           // y (softmax output)
    uint32_t grad_cb_id,        // grad (upstream gradient)
    uint32_t sum_reduce_cb_id,  // sum(y * grad) - broadcasted scalar
    uint32_t out_cb_id,         // output
    uint32_t y_tile_idx,
    uint32_t grad_tile_idx) {
    tile_regs_acquire();

    // Step 1: Compute grad - sum(y * grad) and store in DST[0]
    ckernel::sub_bcast_cols_init_with_dt(grad_cb_id, sum_reduce_cb_id);
    sub_tiles_bcast<BROADCAST_TYPE>(grad_cb_id, sum_reduce_cb_id, grad_tile_idx, 0, DST_REG_ID);

    // Step 2: Multiply y * DST[0], reusing the DST register
#if defined(FP32_DEST_ACC_EN)
    ckernel::reconfig_data_format_srca(y_cb_id);
#endif
    mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(y_cb_id);
    mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        y_cb_id, y_tile_idx, DST_REG_ID);

    tile_regs_commit();
    pack_and_push(DST_REG_ID, out_cb_id);
}

#endif

void kernel_main() {
    // Compile time args
    constexpr uint32_t y_cb_id = get_compile_time_arg_val(0);            // softmax_output (y)
    constexpr uint32_t grad_cb_id = get_compile_time_arg_val(1);         // upstream_grad (grad)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);          // output
    constexpr uint32_t sum_reduce_cb_id = get_compile_time_arg_val(3);   // per-row scalar sum(y * grad)
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(4);         // ones vector for matmul reduction
    constexpr uint32_t partial_cb_id = get_compile_time_arg_val(5);      // 1-tile partial (32 col-partials per row)
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(6);  // width in tiles
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(7);  // block size - must match reader/writer kernels
    constexpr bool full_row_in_l1 = (num_tiles_per_row == tiles_per_block);  // skip second read when full row fits

    // Runtime args
    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // Number of rows to process

    // Initialize compute operations.
    compute_kernel_hw_startup(y_cb_id, grad_cb_id, out_cb_id);
#if !defined(ARCH_WORMHOLE) && !defined(ARCH_BLACKHOLE)
    cb_wait_front(ones_cb_id, ONE_TILE);
#endif

    // Two-pass streaming algorithm for minimal L1 memory
    for (uint32_t row = 0; row < num_rows; ++row) {
#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
        // === PASS 1: sum(y * grad) per row entirely in FP32 DEST ===
        mul_reduce_row_to_scalar<full_row_in_l1>(
            y_cb_id, grad_cb_id, sum_reduce_cb_id, num_tiles_per_row, tiles_per_block);
        // sfpu_reduce_init changes SFPU state, so configure the pass-2 column broadcast afterward.
        sfpu_sub_bcast_col_init();
#else
        // === PASS 1: Quasar fallback through the existing FPU reduction ===
        mul_accumulate_row_to_dst<full_row_in_l1>(
            y_cb_id, grad_cb_id, partial_cb_id, num_tiles_per_row, tiles_per_block);
        reduce_partial_to_scalar(partial_cb_id, ones_cb_id, sum_reduce_cb_id);
#endif

        // === PASS 2: Compute final output (reuse data when full row in L1, else fresh read) ===
        cb_wait_front(sum_reduce_cb_id, ONE_TILE);

        for (uint32_t block_start = 0; block_start < num_tiles_per_row; block_start += tiles_per_block) {
            if constexpr (!full_row_in_l1) {
                cb_wait_front(y_cb_id, tiles_per_block);
                cb_wait_front(grad_cb_id, tiles_per_block);
            }

            // Process each tile: compute y * (grad - sum).
            for (uint32_t i = 0; i < tiles_per_block; ++i) {
#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
                compute_output_tile(y_cb_id, grad_cb_id, sum_reduce_cb_id, out_cb_id, i);
#else
                fused_sub_mul(
                    y_cb_id,           // y
                    grad_cb_id,        // grad
                    sum_reduce_cb_id,  // sum(y * grad)
                    out_cb_id,         // output
                    i,                 // y tile index (relative to block)
                    i);                // grad tile index (relative to block)
#endif
            }

            cb_pop_front(y_cb_id, tiles_per_block);
            cb_pop_front(grad_cb_id, tiles_per_block);
        }

        // Pop sum for this row
        cb_pop_front(sum_reduce_cb_id, ONE_TILE);
    }
}
