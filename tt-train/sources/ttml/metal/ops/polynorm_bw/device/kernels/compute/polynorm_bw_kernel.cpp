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
constexpr auto cb_input_pass_4 = tt::CBIndex::c_3;
constexpr auto cb_input_pass_5 = tt::CBIndex::c_4;
constexpr auto cb_input_pass_6 = tt::CBIndex::c_5;
constexpr auto cb_dL_dout_pass_1 = tt::CBIndex::c_6;
constexpr auto cb_dL_dout_pass_2 = tt::CBIndex::c_7;
constexpr auto cb_dL_dout_pass_3 = tt::CBIndex::c_8;
constexpr auto cb_dL_dout_pass_4 = tt::CBIndex::c_9;
constexpr auto cb_scaler = tt::CBIndex::c_10;
constexpr auto cb_eps = tt::CBIndex::c_11;
constexpr auto cb_one = tt::CBIndex::c_12;
constexpr auto cb_w0 = tt::CBIndex::c_13;
constexpr auto cb_w1 = tt::CBIndex::c_14;
constexpr auto cb_w2 = tt::CBIndex::c_15;
// CBs with intermediate computations
constexpr auto cb_sum_x2 = tt::CBIndex::c_16;
constexpr auto cb_sum_x4 = tt::CBIndex::c_17;
constexpr auto cb_sum_x6 = tt::CBIndex::c_18;
constexpr auto cb_sum_scale_1 = tt::CBIndex::c_19;
constexpr auto cb_sum_scale_2 = tt::CBIndex::c_20;
constexpr auto cb_sum_scale_3 = tt::CBIndex::c_21;
constexpr auto cb_inv_rms_x = tt::CBIndex::c_22;
constexpr auto cb_inv_rms_x2 = tt::CBIndex::c_23;
constexpr auto cb_inv_rms_x3 = tt::CBIndex::c_24;
constexpr auto cb_coeff_1 = tt::CBIndex::c_25;
constexpr auto cb_coeff_2 = tt::CBIndex::c_26;
constexpr auto cb_coeff_3 = tt::CBIndex::c_27;
// CB with output data
constexpr auto cb_output = tt::CBIndex::c_28;

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

void reduce_sum_to_scalar(const uint32_t cb_sum, const uint32_t cb_scalar) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_one, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;
    reconfig_data_format(cb_sum, cb_one);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, cb_scalar);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_one, 0, 0, reg_acc);
    reduce_uninit();
    tile_regs_commit();
    pack_and_push(reg_acc, cb_scalar);
    cb_pop_front(cb_sum, onetile);
}

void compute_coeff_tile(const uint32_t cb_inv, const uint32_t cb_scale, const uint32_t cb_coeff) {
    cb_wait_front(cb_inv, onetile);
    cb_wait_front(cb_scale, onetile);
    cb_wait_front(cb_scaler, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_inv = 0U;
    constexpr uint32_t reg_scale = 1U;
    constexpr uint32_t reg_scaler = 2U;
    unary_bcast_init<BroadcastType::COL>(cb_inv, cb_inv);
    unary_bcast<BroadcastType::COL>(cb_inv, 0, reg_inv);
    unary_bcast_init<BroadcastType::COL>(cb_scale, cb_scale);
    unary_bcast<BroadcastType::COL>(cb_scale, 0, reg_scale);
    copy_tile_init(cb_scaler);
    copy_tile(cb_scaler, 0, reg_scaler);
    mul_binary_tile_init();
    mul_binary_tile(reg_inv, reg_inv, reg_inv);      // inv^2
    mul_binary_tile(reg_inv, reg_scale, reg_scale);  // inv^2 * scale
    unary_bcast_init<BroadcastType::COL>(cb_inv, cb_inv);
    unary_bcast<BroadcastType::COL>(cb_inv, 0, reg_inv);
    mul_binary_tile(reg_scale, reg_inv, reg_scale);     // inv^3 * scale
    mul_binary_tile(reg_scale, reg_scaler, reg_scale);  // inv^3 * scale / C
    tile_regs_commit();
    pack_and_push(reg_scale, cb_coeff);
    cb_pop_front(cb_scale, onetile);
}

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
                mul_binary_tile(reg_x, reg_x, reg_sum_x2);
                mul_binary_tile(reg_sum_x2, reg_x, reg_tmp);
                mul_binary_tile(reg_tmp, reg_tmp, reg_sum_x6);
                first_tile = false;
                continue;
            }
            mul_binary_tile(reg_x, reg_x, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum_x2, reg_tmp, reg_sum_x2);
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);
            mul_binary_tile(reg_tmp, reg_tmp, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum_x6, reg_tmp, reg_sum_x6);
        }
        cb_pop_front(cb_input_pass_1, block_size);
    }
    cb_reserve_back(cb_sum_x2, onetile);
    cb_reserve_back(cb_sum_x6, onetile);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_sum_x2);
    pack_tile(reg_sum_x2, cb_sum_x2);
    pack_reconfig_data_format(cb_sum_x6);
    pack_tile(reg_sum_x6, cb_sum_x6);
    tile_regs_release();
    cb_push_back(cb_sum_x2, onetile);
    cb_push_back(cb_sum_x6, onetile);
}

void accumulate_sum_x4_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_x2 = 1U;
    constexpr uint32_t reg_sum_x4 = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_2, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_2);
            copy_tile(cb_input_pass_2, block_idx, reg_x);
            mul_binary_tile_init();
            if (first_tile) {
                mul_binary_tile(reg_x, reg_x, reg_x2);
                mul_binary_tile(reg_x2, reg_x2, reg_sum_x4);
                first_tile = false;
                continue;
            }
            mul_binary_tile(reg_x, reg_x, reg_x2);
            mul_binary_tile(reg_x2, reg_x2, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum_x4, reg_tmp, reg_sum_x4);
        }
        cb_pop_front(cb_input_pass_2, block_size);
    }
    tile_regs_commit();
    pack_and_push(reg_sum_x4, cb_sum_x4);
}

void accumulate_scale1_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_dout = 1U;
    constexpr uint32_t reg_sum = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_3, block_size);
        cb_wait_front(cb_dL_dout_pass_1, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_3);
            copy_tile(cb_input_pass_3, block_idx, reg_x);
            copy_tile_init(cb_dL_dout_pass_1);
            copy_tile(cb_dL_dout_pass_1, block_idx, reg_dout);
            copy_tile_init(cb_w2);
            copy_tile(cb_w2, 0, reg_tmp);
            mul_binary_tile_init();
            mul_binary_tile(reg_dout, reg_tmp, reg_tmp);  // dout * w2
            if (first_tile) {
                mul_binary_tile(reg_x, reg_tmp, reg_sum);
                first_tile = false;
                continue;
            }
            mul_binary_tile(reg_x, reg_tmp, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum, reg_tmp, reg_sum);
        }
        cb_pop_front(cb_input_pass_3, block_size);
        cb_pop_front(cb_dL_dout_pass_1, block_size);
    }
    tile_regs_commit();
    pack_and_push(reg_sum, cb_sum_scale_1);
}

void accumulate_scale2_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_dout = 1U;
    constexpr uint32_t reg_sum = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_4, block_size);
        cb_wait_front(cb_dL_dout_pass_2, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_4);
            copy_tile(cb_input_pass_4, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);  // x^2
            copy_tile_init(cb_dL_dout_pass_2);
            copy_tile(cb_dL_dout_pass_2, block_idx, reg_dout);
            copy_tile_init(cb_w1);
            copy_tile(cb_w1, 0, reg_x);
            mul_binary_tile(reg_dout, reg_x, reg_dout);  // dout * w1
            if (first_tile) {
                mul_binary_tile(reg_tmp, reg_dout, reg_sum);
                first_tile = false;
                continue;
            }
            mul_binary_tile(reg_tmp, reg_dout, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum, reg_tmp, reg_sum);
        }
        cb_pop_front(cb_input_pass_4, block_size);
        cb_pop_front(cb_dL_dout_pass_2, block_size);
    }
    tile_regs_commit();
    pack_and_push(reg_sum, cb_sum_scale_2);
}

void accumulate_scale3_for_row() {
    constexpr uint32_t reg_x = 0U;
    constexpr uint32_t reg_dout = 1U;
    constexpr uint32_t reg_sum = 2U;
    constexpr uint32_t reg_tmp = 3U;

    bool first_tile = true;
    tile_regs_acquire();
    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_5, block_size);
        cb_wait_front(cb_dL_dout_pass_3, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            copy_tile_init(cb_input_pass_5);
            copy_tile(cb_input_pass_5, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);    // x^2
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);  // x^3
            copy_tile_init(cb_dL_dout_pass_3);
            copy_tile(cb_dL_dout_pass_3, block_idx, reg_dout);
            copy_tile_init(cb_w0);
            copy_tile(cb_w0, 0, reg_x);
            mul_binary_tile(reg_dout, reg_x, reg_dout);  // dout * w0
            if (first_tile) {
                mul_binary_tile(reg_tmp, reg_dout, reg_sum);
                first_tile = false;
                continue;
            }
            mul_binary_tile(reg_tmp, reg_dout, reg_tmp);
            add_binary_tile_init();
            add_binary_tile(reg_sum, reg_tmp, reg_sum);
        }
        cb_pop_front(cb_input_pass_5, block_size);
        cb_pop_front(cb_dL_dout_pass_3, block_size);
    }
    tile_regs_commit();
    pack_and_push(reg_sum, cb_sum_scale_3);
}

void emit_output_for_row() {
    constexpr uint32_t reg0 = 0U;
    constexpr uint32_t reg1 = 1U;
    constexpr uint32_t reg_acc = 2U;
    constexpr uint32_t reg_tmp = 3U;

    cb_wait_front(cb_inv_rms_x, onetile);
    cb_wait_front(cb_inv_rms_x2, onetile);
    cb_wait_front(cb_inv_rms_x3, onetile);
    cb_wait_front(cb_coeff_1, onetile);
    cb_wait_front(cb_coeff_2, onetile);
    cb_wait_front(cb_coeff_3, onetile);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_6, block_size);
        cb_wait_front(cb_dL_dout_pass_4, block_size);
        cb_reserve_back(cb_output, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg0);  // x
            copy_tile_init(cb_dL_dout_pass_4);
            copy_tile(cb_dL_dout_pass_4, block_idx, reg1);  // dout

            // term1: dL/dx from normalized x
            copy_tile_init(cb_w2);
            copy_tile(cb_w2, 0, reg_tmp);
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg_tmp, reg_tmp);  // dout*w2
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, reg1);
            mul_binary_tile(reg_tmp, reg1, reg_acc);  // dout*w2*inv
            unary_bcast_init<BroadcastType::COL>(cb_coeff_1, cb_coeff_1);
            unary_bcast<BroadcastType::COL>(cb_coeff_1, 0, reg1);
            mul_binary_tile(reg0, reg1, reg_tmp);  // x*coeff1
            sub_binary_tile_init();
            sub_binary_tile(reg_acc, reg_tmp, reg_acc);

            // term2: dL/dx via x^2 branch
            copy_tile_init(cb_dL_dout_pass_4);
            copy_tile(cb_dL_dout_pass_4, block_idx, reg1);
            copy_tile_init(cb_w1);
            copy_tile(cb_w1, 0, reg_tmp);
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg_tmp, reg_tmp);  // dout*w1
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x2, cb_inv_rms_x2);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x2, 0, reg1);
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // dout*w1*inv2
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg1);  // x
            mul_binary_tile(reg1, reg1, reg1);            // x^2
            unary_bcast_init<BroadcastType::COL>(cb_coeff_2, cb_coeff_2);
            unary_bcast<BroadcastType::COL>(cb_coeff_2, 0, reg0);
            mul_binary_tile(reg1, reg0, reg0);  // x^2*coeff2
            sub_binary_tile_init();
            sub_binary_tile(reg_tmp, reg0, reg_tmp);  // g2
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg1);  // x
            add_binary_tile_init();
            add_binary_tile(reg1, reg1, reg1);  // 2x
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // term2
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // term3: dL/dx via x^3 branch
            copy_tile_init(cb_dL_dout_pass_4);
            copy_tile(cb_dL_dout_pass_4, block_idx, reg1);
            copy_tile_init(cb_w0);
            copy_tile(cb_w0, 0, reg_tmp);
            mul_binary_tile_init();
            mul_binary_tile(reg1, reg_tmp, reg_tmp);  // dout*w0
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x3, cb_inv_rms_x3);
            unary_bcast<BroadcastType::COL>(cb_inv_rms_x3, 0, reg1);
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // dout*w0*inv3
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg1);  // x
            mul_binary_tile(reg1, reg1, reg1);            // x^2
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg0);  // x
            mul_binary_tile(reg1, reg0, reg0);            // x^3
            unary_bcast_init<BroadcastType::COL>(cb_coeff_3, cb_coeff_3);
            unary_bcast<BroadcastType::COL>(cb_coeff_3, 0, reg1);
            mul_binary_tile(reg0, reg1, reg0);  // x^3 * coeff3
            sub_binary_tile_init();
            sub_binary_tile(reg_tmp, reg0, reg_tmp);  // g3
            copy_tile_init(cb_input_pass_6);
            copy_tile(cb_input_pass_6, block_idx, reg1);
            mul_binary_tile(reg1, reg1, reg1);  // x^2
            add_binary_tile_init();
            add_binary_tile(reg1, reg1, reg0);  // 2x^2
            add_binary_tile(reg0, reg1, reg1);  // 3x^2
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg1, reg_tmp);  // term3
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            tile_regs_commit();
            pack_l1_acc_block(cb_output, true, 1U, block_idx);
        }
        cb_push_back(cb_output, block_size);
        cb_pop_front(cb_input_pass_6, block_size);
        cb_pop_front(cb_dL_dout_pass_4, block_size);
    }

    cb_pop_front(cb_inv_rms_x, onetile);
    cb_pop_front(cb_inv_rms_x2, onetile);
    cb_pop_front(cb_inv_rms_x3, onetile);
    cb_pop_front(cb_coeff_1, onetile);
    cb_pop_front(cb_coeff_2, onetile);
    cb_pop_front(cb_coeff_3, onetile);
}

void kernel_main() {
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_w0, onetile);
    cb_wait_front(cb_w1, onetile);
    cb_wait_front(cb_w2, onetile);

    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        (void)row;
        accumulate_sum_x2_and_x6_for_row();
        accumulate_sum_x4_for_row();
        accumulate_scale1_for_row();
        accumulate_scale2_for_row();
        accumulate_scale3_for_row();

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);
        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);
        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);
        reduce_sum_to_scalar(cb_sum_scale_1, cb_sum_scale_1);
        reduce_sum_to_scalar(cb_sum_scale_2, cb_sum_scale_2);
        reduce_sum_to_scalar(cb_sum_scale_3, cb_sum_scale_3);

        compute_coeff_tile(cb_inv_rms_x, cb_sum_scale_1, cb_coeff_1);
        compute_coeff_tile(cb_inv_rms_x2, cb_sum_scale_2, cb_coeff_2);
        compute_coeff_tile(cb_inv_rms_x3, cb_sum_scale_3, cb_coeff_3);

        emit_output_for_row();
    }

    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_w0, onetile);
    cb_pop_front(cb_w1, onetile);
    cb_pop_front(cb_w2, onetile);
}
