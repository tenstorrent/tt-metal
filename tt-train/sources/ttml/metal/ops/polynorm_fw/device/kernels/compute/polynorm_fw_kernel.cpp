// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t num_inner = get_compile_time_arg_val(2);

// CBs with input data / scalar parameters
constexpr auto cb_input_pass_1 = tt::CBIndex::c_0;
constexpr auto cb_input_pass_2 = tt::CBIndex::c_1;
constexpr auto cb_input_pass_3 = tt::CBIndex::c_2;
constexpr auto cb_scaler = tt::CBIndex::c_3;
constexpr auto cb_eps = tt::CBIndex::c_4;
constexpr auto cb_w0 = tt::CBIndex::c_5;
constexpr auto cb_w1 = tt::CBIndex::c_6;
constexpr auto cb_w2 = tt::CBIndex::c_7;
constexpr auto cb_bias = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr auto cb_sum_x2 = tt::CBIndex::c_9;
constexpr auto cb_sum_x4 = tt::CBIndex::c_10;
constexpr auto cb_sum_x6 = tt::CBIndex::c_11;
constexpr auto cb_inv_rms_x = tt::CBIndex::c_12;
constexpr auto cb_inv_rms_x2 = tt::CBIndex::c_13;
constexpr auto cb_inv_rms_x3 = tt::CBIndex::c_14;
// CB with output data
constexpr auto cb_output = tt::CBIndex::c_15;

// Reduce a row-sum tile into inv_rms = 1 / sqrt(sum * scaler + eps).
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_eps = 1U;
    reconfig_data_format(cb_sum, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, 0, 0, reg_acc);
    reduce_uninit();

    reconfig_data_format_srca(cb_eps);
    copy_tile_init(cb_eps);
    copy_tile(cb_eps, 0, reg_eps);

    reconfig_data_format(cb_sum, cb_eps);
    add_binary_tile_init();
    add_binary_tile(reg_acc, reg_eps, reg_acc);

    sqrt_tile_init();
    sqrt_tile(reg_acc);
    recip_tile_init();
    recip_tile(reg_acc);

    tile_regs_commit();
    pack_and_push(reg_acc, cb_inv_rms);
    cb_pop_front(cb_sum, onetile);
}

// Pass-1a: compute and emit row sums for x^2 and x^6.
void accumulate_sum_x2_and_x6_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_tmp = 1U;
    constexpr uint32_t reg_sum_x2 = 2U;
    constexpr uint32_t reg_sum_x6 = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_1, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_1);
            copy_tile(cb_input_pass_1, block_idx, reg_x);
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
        cb_pop_front(cb_input_pass_1, block_size);
    }
    tile_regs_commit();
    pack_and_push_two_tiles(reg_sum_x2, cb_sum_x2, reg_sum_x6, cb_sum_x6);
}

// Pass-1b: compute and emit row sum for x^4.
void accumulate_sum_x4_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_x2 = 1U;
    constexpr uint32_t reg_sum_x4 = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_3, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_3);
            copy_tile(cb_input_pass_3, block_idx, reg_x);
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
        cb_pop_front(cb_input_pass_3, block_size);
    }
    tile_regs_commit();
    pack_and_push(reg_sum_x4, cb_sum_x4);
}

// Pass-2: compute weighted normalized terms and write final PolyNorm output.
void emit_output_for_row() {
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_x = 1U;
    constexpr uint32_t reg_tmp = 2U;
    constexpr uint32_t reg_bcast_or_scalar = 3U;

    cb_wait_front(cb_inv_rms_x, onetile);
    cb_wait_front(cb_inv_rms_x2, onetile);
    cb_wait_front(cb_inv_rms_x3, onetile);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_2, block_size);
        cb_reserve_back(cb_output, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();

            // term3 = w0 * (x^3 * inv_rms_x3)
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x3, cb_inv_rms_x3);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x3, 0, reg_bcast_or_scalar);
            copy_tile_init(cb_input_pass_2);
            copy_tile(cb_input_pass_2, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);    // x^2
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);  // x^3
            reconfig_data_format(cb_input_pass_2, cb_inv_rms_x3);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^3)
            copy_tile_init(cb_w0);
            copy_tile(cb_w0, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_acc);  // accumulator init

            // term2 = w1 * (x^2 * inv_rms_x2)
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x2, cb_inv_rms_x2);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x2, 0, reg_bcast_or_scalar);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);  // x^2
            reconfig_data_format(cb_input_pass_2, cb_inv_rms_x2);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^2)
            copy_tile_init(cb_w1);
            copy_tile(cb_w1, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term2
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // term1 = w2 * (x * inv_rms_x)
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, reg_bcast_or_scalar);
            reconfig_data_format(cb_input_pass_2, cb_inv_rms_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_bcast_or_scalar, reg_tmp);  // norm(x)
            copy_tile_init(cb_w2);
            copy_tile(cb_w2, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term1
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // Add bias and emit output tile.
            copy_tile_init(cb_bias);
            copy_tile(cb_bias, 0, reg_bcast_or_scalar);
            add_binary_tile(reg_acc, reg_bcast_or_scalar, reg_acc);

            tile_regs_commit();
            // Keep reserve(block_size)/push(block_size) invariant, but only fill current_block_size slots.
            pack_l1_acc_block(cb_output, /* first_block */ true, /* num_tiles */ 1U, /* dst_start_index */ block_idx);
        }

        cb_push_back(cb_output, block_size);
        cb_pop_front(cb_input_pass_2, block_size);
    }

    cb_pop_front(cb_inv_rms_x, onetile);
    cb_pop_front(cb_inv_rms_x2, onetile);
    cb_pop_front(cb_inv_rms_x3, onetile);
}

// Compute PolyNorm forward for all rows assigned to this core.
void kernel_main() {
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_w0, onetile);
    cb_wait_front(cb_w1, onetile);
    cb_wait_front(cb_w2, onetile);
    cb_wait_front(cb_bias, onetile);

    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        (void)row;
        accumulate_sum_x2_and_x6_for_row();
        accumulate_sum_x4_for_row();

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);
        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);
        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);

        emit_output_for_row();
    }

    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);
    cb_pop_front(cb_w0, onetile);
    cb_pop_front(cb_w1, onetile);
    cb_pop_front(cb_w2, onetile);
    cb_pop_front(cb_bias, onetile);
}
