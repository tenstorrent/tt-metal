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
constexpr auto kCbOnes = tt::CBIndex::c_15;
constexpr auto kCbZero = tt::CBIndex::c_17;

constexpr uint32_t kOneTile = 1U;

// Reduce a row-sum tile into inv_rms = 1 / sqrt(sum * scaler + eps).
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, kOneTile);
    cb_wait_front(kCbScaler, kOneTile);
    cb_wait_front(kCbEps, kOneTile);

    tile_regs_acquire();
    constexpr uint32_t kAccReg = 0U;
    constexpr uint32_t kEpsReg = 1U;
    reconfig_data_format(cb_sum, kCbScaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, kCbScaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, kCbScaler, 0, 0, kAccReg);
    reduce_uninit();

    reconfig_data_format_srca(kCbEps);
    copy_tile_init(kCbEps);
    copy_tile(kCbEps, 0, kEpsReg);

    reconfig_data_format(cb_sum, kCbEps);
    add_binary_tile_init();
    add_binary_tile(kAccReg, kEpsReg, kAccReg);

    sqrt_tile_init();
    sqrt_tile(kAccReg);
    recip_tile_init();
    recip_tile(kAccReg);

    tile_regs_commit();
    pack_and_push(kAccReg, cb_inv_rms);
    cb_pop_front(cb_sum, kOneTile);
}

// Pass-1a: compute and emit row sums for x^2 and x^6.
void accumulate_sum_x2_and_x6_for_row() {
    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass1, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            if (first_tile) {
                copy_tile_init(kCbInputPass1);
                copy_tile(kCbInputPass1, block_idx, 1U);  // x
                mul_binary_tile_init();
                mul_binary_tile(1U, 1U, 0U);  // x^2 accumulator
                mul_binary_tile(1U, 1U, 3U);  // x^2
                mul_binary_tile(3U, 1U, 3U);  // x^3
                mul_binary_tile(3U, 3U, 2U);  // x^6 accumulator
                first_tile = false;
            } else {
                copy_tile_init(kCbInputPass1);
                copy_tile(kCbInputPass1, block_idx, 1U);  // x
                mul_binary_tile_init();
                mul_binary_tile(1U, 1U, 4U);  // x^2
                add_binary_tile_init();
                add_binary_tile(0U, 4U, 0U);
                mul_binary_tile(4U, 1U, 3U);  // x^3
                mul_binary_tile(3U, 3U, 5U);  // x^6
                add_binary_tile_init();
                add_binary_tile(2U, 5U, 2U);
            }
        }
        cb_pop_front(kCbInputPass1, kBlockSize);
    }
    cb_reserve_back(kCbSumX2, kOneTile);
    cb_reserve_back(kCbSumX6, kOneTile);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(kCbSumX2);
    pack_tile(0U, kCbSumX2);
    pack_reconfig_data_format(kCbSumX6);
    pack_tile(2U, kCbSumX6);
    tile_regs_release();
    cb_push_back(kCbSumX2, kOneTile);
    cb_push_back(kCbSumX6, kOneTile);
}

// Pass-1b: compute and emit row sum for x^4.
void accumulate_sum_x4_for_row() {
    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass3, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            if (first_tile) {
                copy_tile_init(kCbInputPass3);
                copy_tile(kCbInputPass3, block_idx, 1U);  // x
                mul_binary_tile_init();
                mul_binary_tile(1U, 1U, 1U);  // x^2
                mul_binary_tile(1U, 1U, 0U);  // x^4 accumulator
                first_tile = false;
            } else {
                copy_tile_init(kCbInputPass3);
                copy_tile(kCbInputPass3, block_idx, 1U);  // x
                mul_binary_tile_init();
                mul_binary_tile(1U, 1U, 1U);  // x^2
                mul_binary_tile(1U, 1U, 3U);  // x^4
                add_binary_tile_init();
                add_binary_tile(0U, 3U, 0U);
            }
        }
        cb_pop_front(kCbInputPass3, kBlockSize);
    }
    tile_regs_commit();
    pack_and_push(0U, kCbSumX4);
}

// Pass-2: compute weighted normalized terms and write final PolyNorm output.
void emit_output_for_row() {
    cb_wait_front(kCbInvRmsX, kOneTile);
    cb_wait_front(kCbInvRmsX2, kOneTile);
    cb_wait_front(kCbInvRmsX3, kOneTile);

    for (uint32_t col = 0; col < kNumInner; col += kBlockSize) {
        const uint32_t current_block_size = (col + kBlockSize <= kNumInner) ? kBlockSize : (kNumInner - col);
        cb_wait_front(kCbInputPass2, kBlockSize);
        cb_reserve_back(kCbOutput, kBlockSize);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX3, kCbInvRmsX3);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX3, 0, 5U);
            copy_tile_init(kCbInputPass2);
            copy_tile(kCbInputPass2, block_idx, 0U);  // x
            mul_binary_tile_init();
            mul_binary_tile(0U, 0U, 1U);  // x^2
            mul_binary_tile(1U, 0U, 2U);  // x^3
            reconfig_data_format(kCbInputPass2, kCbInvRmsX3);
            mul_binary_tile_init();
            mul_binary_tile(2U, 5U, 2U);  // norm(x^3)
            copy_tile_init(kCbW0);
            copy_tile(kCbW0, 0, 4U);
            mul_binary_tile(2U, 4U, 2U);  // w0 * norm(x^3)
            tile_regs_commit();
            pack_and_push(2U, kCbSumX2);

            tile_regs_acquire();
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX2, kCbInvRmsX2);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX2, 0, 1U);
            copy_tile_init(kCbInputPass2);
            copy_tile(kCbInputPass2, block_idx, 0U);  // x
            mul_binary_tile_init();
            mul_binary_tile(0U, 0U, 0U);  // x^2
            reconfig_data_format(kCbInputPass2, kCbInvRmsX2);
            mul_binary_tile_init();
            mul_binary_tile(0U, 1U, 0U);  // norm(x^2)
            copy_tile_init(kCbW1);
            copy_tile(kCbW1, 0, 4U);
            mul_binary_tile(0U, 4U, 0U);  // w1 * norm(x^2)
            tile_regs_commit();
            pack_and_push(0U, kCbSumX4);

            tile_regs_acquire();
            unary_bcast_init<BroadcastType::COL>(kCbInvRmsX, kCbInvRmsX);
            unary_bcast<BroadcastType::COL>(kCbInvRmsX, 0, 5U);
            copy_tile_init(kCbInputPass2);
            copy_tile(kCbInputPass2, block_idx, 0U);  // x
            reconfig_data_format(kCbInputPass2, kCbInvRmsX);
            mul_binary_tile_init();
            mul_binary_tile(0U, 5U, 0U);  // norm(x)
            copy_tile_init(kCbW2);
            copy_tile(kCbW2, 0, 4U);
            mul_binary_tile(0U, 4U, 0U);  // w2 * norm(x)
            tile_regs_commit();
            pack_and_push(0U, kCbSumX6);

            cb_wait_front(kCbSumX2, kOneTile);
            cb_wait_front(kCbSumX4, kOneTile);
            cb_wait_front(kCbSumX6, kOneTile);
            tile_regs_acquire();
            copy_tile_init(kCbSumX2);
            copy_tile(kCbSumX2, 0, 2U);
            copy_tile_init(kCbSumX4);
            copy_tile(kCbSumX4, 0, 1U);
            add_binary_tile_init();
            add_binary_tile(2U, 1U, 2U);
            copy_tile_init(kCbSumX6);
            copy_tile(kCbSumX6, 0, 0U);
            add_binary_tile(2U, 0U, 2U);
            copy_tile_init(kCbBias);
            copy_tile(kCbBias, 0, 4U);
            add_binary_tile(2U, 4U, 2U);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(kCbOutput);
            pack_tile(2U, kCbOutput);
            tile_regs_release();
            cb_pop_front(kCbSumX2, kOneTile);
            cb_pop_front(kCbSumX4, kOneTile);
            cb_pop_front(kCbSumX6, kOneTile);
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
    cb_wait_front(kCbOnes, kOneTile);
    cb_wait_front(kCbZero, kOneTile);

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
    cb_pop_front(kCbOnes, kOneTile);
    cb_pop_front(kCbZero, kOneTile);
}
