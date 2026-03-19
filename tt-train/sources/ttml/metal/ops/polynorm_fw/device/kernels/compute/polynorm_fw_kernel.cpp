// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t kNumRowsPerCore = get_compile_time_arg_val(0);
constexpr uint32_t kBlockSize = get_compile_time_arg_val(1);
constexpr uint32_t kNumInner = get_compile_time_arg_val(2);

constexpr auto kCbInputPass1 = tt::CBIndex::c_0;
constexpr auto kCbInputPass2 = tt::CBIndex::c_1;
constexpr auto kCbInputPass3 = tt::CBIndex::c_19;
constexpr auto kCbScaler = tt::CBIndex::c_2;
constexpr auto kCbEps = tt::CBIndex::c_3;
constexpr auto kCbW0 = tt::CBIndex::c_4;
constexpr auto kCbW1 = tt::CBIndex::c_5;
constexpr auto kCbW2 = tt::CBIndex::c_6;
constexpr auto kCbBias = tt::CBIndex::c_7;
constexpr auto kCbSumX2 = tt::CBIndex::c_8;
constexpr auto kCbSumX4 = tt::CBIndex::c_9;
constexpr auto kCbSumX6 = tt::CBIndex::c_10;
constexpr auto kCbInvRmsX = tt::CBIndex::c_11;
constexpr auto kCbInvRmsX2 = tt::CBIndex::c_12;
constexpr auto kCbInvRmsX3 = tt::CBIndex::c_13;
constexpr auto kCbOutput = tt::CBIndex::c_14;

constexpr uint32_t kOneTile = 1U;

// Reduce a row-sum tile into inv_rms = 1 / sqrt(sum * scaler + eps).
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, kOneTile);
    cb_wait_front(kCbScaler, kOneTile);
    cb_wait_front(kCbEps, kOneTile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_eps = 1U;
    reconfig_data_format(cb_sum, kCbScaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, kCbScaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, kCbScaler, 0, 0, reg_acc);
    reduce_uninit();

    reconfig_data_format_srca(kCbEps);
    copy_tile_init(kCbEps);
    copy_tile(kCbEps, 0, reg_eps);

    reconfig_data_format(cb_sum, kCbEps);
    add_binary_tile_init();
    add_binary_tile(reg_acc, reg_eps, reg_acc);

    sqrt_tile_init();
    sqrt_tile(reg_acc);
    recip_tile_init();
    recip_tile(reg_acc);

    tile_regs_commit();
    pack_and_push(reg_acc, cb_inv_rms);
    cb_pop_front(cb_sum, kOneTile);
}

// Pass-1a: compute and emit row sums for x^2 and x^6.
void accumulate_sum_x2_and_x6_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_tmp = 1U;
    constexpr uint32_t reg_sum_x2 = 2U;
    constexpr uint32_t reg_sum_x6 = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass1, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(kCbInputPass1);
            copy_tile(kCbInputPass1, block_idx, reg_x);
            mul_binary_tile_init();
            if (first_tile) {
                mul_binary_tile(reg_x, reg_x, reg_sum_x2);      // x^2 sum init
                mul_binary_tile(reg_sum_x2, reg_x, reg_tmp);    // x^3
                mul_binary_tile(reg_tmp, reg_tmp, reg_sum_x6);  // x^6 sum init
                first_tile = false;
                continue;
            }

            mul_binary_tile(reg_x, reg_x, reg_tmp);  // x^2
            add_binary_tile_init();
            add_binary_tile(reg_sum_x2, reg_tmp, reg_sum_x2);  // sum(x^2)
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);          // x^3
            mul_binary_tile(reg_tmp, reg_tmp, reg_tmp);        // x^6
            add_binary_tile_init();
            add_binary_tile(reg_sum_x6, reg_tmp, reg_sum_x6);  // sum(x^6)
        }
        cb_pop_front(kCbInputPass1, kBlockSize);
    }
    cb_reserve_back(kCbSumX2, kOneTile);
    cb_reserve_back(kCbSumX6, kOneTile);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kCbSumX2);
    pack_tile(reg_sum_x2, kCbSumX2);
    pack_reconfig_data_format(kCbSumX6);
    pack_tile(reg_sum_x6, kCbSumX6);
    tile_regs_release();
    cb_push_back(kCbSumX2, kOneTile);
    cb_push_back(kCbSumX6, kOneTile);
}

// Pass-1b: compute and emit row sum for x^4.
void accumulate_sum_x4_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_x2 = 1U;
    constexpr uint32_t reg_sum_x4 = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass3, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(kCbInputPass3);
            copy_tile(kCbInputPass3, block_idx, reg_x);
            mul_binary_tile_init();
            if (first_tile) {
                mul_binary_tile(reg_x, reg_x, reg_x2);        // x^2
                mul_binary_tile(reg_x2, reg_x2, reg_sum_x4);  // x^4 sum init
                first_tile = false;
                continue;
            }

            mul_binary_tile(reg_x, reg_x, reg_x2);     // x^2
            mul_binary_tile(reg_x2, reg_x2, reg_tmp);  // x^4
            add_binary_tile_init();
            add_binary_tile(reg_sum_x4, reg_tmp, reg_sum_x4);  // sum(x^4)
        }
        cb_pop_front(kCbInputPass3, kBlockSize);
    }
    tile_regs_commit();
    pack_and_push(reg_sum_x4, kCbSumX4);
}

// Pass-2: compute weighted normalized terms and write final PolyNorm output.
void emit_output_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_acc = 1U;
    constexpr uint32_t reg_tmp = 2U;
    constexpr uint32_t reg_bcast_or_scalar = 3U;

    cb_wait_front(kCbInvRmsX, kOneTile);
    cb_wait_front(kCbInvRmsX2, kOneTile);
    cb_wait_front(kCbInvRmsX3, kOneTile);

    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass2, kBlockSize);
        cb_reserve_back(kCbOutput, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();

            // term3 = w0 * (x^3 * inv_rms_x3)
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX3, kCbInvRmsX3);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX3, 0, reg_bcast_or_scalar);
            copy_tile_init(kCbInputPass2);
            copy_tile(kCbInputPass2, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);    // x^2
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);  // x^3
            reconfig_data_format(kCbInputPass2, kCbInvRmsX3);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^3)
            copy_tile_init(kCbW0);
            copy_tile(kCbW0, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_acc);  // accumulator init

            // term2 = w1 * (x^2 * inv_rms_x2)
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX2, kCbInvRmsX2);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX2, 0, reg_bcast_or_scalar);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);  // x^2
            reconfig_data_format(kCbInputPass2, kCbInvRmsX2);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^2)
            copy_tile_init(kCbW1);
            copy_tile(kCbW1, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term2
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // term1 = w2 * (x * inv_rms_x)
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX, kCbInvRmsX);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX, 0, reg_bcast_or_scalar);
            reconfig_data_format(kCbInputPass2, kCbInvRmsX);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_bcast_or_scalar, reg_tmp);  // norm(x)
            copy_tile_init(kCbW2);
            copy_tile(kCbW2, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term1
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // Add bias and emit output tile.
            copy_tile_init(kCbBias);
            copy_tile(kCbBias, 0, reg_bcast_or_scalar);
            add_binary_tile(reg_acc, reg_bcast_or_scalar, reg_acc);

            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(kCbOutput);
            pack_tile(reg_acc, kCbOutput);
            tile_regs_release();
        }

        cb_push_back(kCbOutput, kBlockSize);
        cb_pop_front(kCbInputPass2, kBlockSize);
    }

    cb_pop_front(kCbInvRmsX, kOneTile);
    cb_pop_front(kCbInvRmsX2, kOneTile);
    cb_pop_front(kCbInvRmsX3, kOneTile);
}

// Compute PolyNorm forward for all rows assigned to this core.
void kernel_main() {
    cb_wait_front(kCbScaler, kOneTile);
    cb_wait_front(kCbEps, kOneTile);
    cb_wait_front(kCbW0, kOneTile);
    cb_wait_front(kCbW1, kOneTile);
    cb_wait_front(kCbW2, kOneTile);
    cb_wait_front(kCbBias, kOneTile);

    init_sfpu(kCbInputPass1, kCbOutput);
    binary_op_init_common(kCbInputPass1, kCbInputPass1, kCbOutput);

    for (uint32_t row = 0; row < kNumRowsPerCore; ++row) {
        (void)row;
        accumulate_sum_x2_and_x6_for_row();
        accumulate_sum_x4_for_row();

        reduce_sum_to_inv_rms(kCbSumX2, kCbInvRmsX);
        reduce_sum_to_inv_rms(kCbSumX4, kCbInvRmsX2);
        reduce_sum_to_inv_rms(kCbSumX6, kCbInvRmsX3);

        emit_output_for_row();
    }

    cb_pop_front(kCbScaler, kOneTile);
    cb_pop_front(kCbEps, kOneTile);
    cb_pop_front(kCbW0, kOneTile);
    cb_pop_front(kCbW1, kOneTile);
    cb_pop_front(kCbW2, kOneTile);
    cb_pop_front(kCbBias, kOneTile);
}
