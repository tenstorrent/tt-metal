// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr auto cb_param_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_exp_avg_idx = tt::CBIndex::c_2;
constexpr auto cb_exp_avg_sq_idx = tt::CBIndex::c_3;
constexpr auto cb_max_exp_avg_sq_in_idx = tt::CBIndex::c_4;

constexpr auto cb_output_idx = tt::CBIndex::c_16;
constexpr auto cb_exp_avg_out_idx = tt::CBIndex::c_17;
constexpr auto cb_exp_avg_sq_out_idx = tt::CBIndex::c_18;
constexpr auto cb_max_exp_avg_sq_out_idx = tt::CBIndex::c_19;

constexpr auto cb_m_t = tt::CBIndex::c_24;
constexpr auto cb_v_t = tt::CBIndex::c_25;
constexpr auto cb_max_exp_avg_sq_idx = tt::CBIndex::c_26;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t twice_block_size = 2 * block_size;

void MAIN {
    constexpr uint32_t fp32_one = 0x3F800000U;  // hexadecimal encoding of 1.0f in uint32_t

    uint32_t runtime_args_counter = 0;
    uint32_t lr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t beta1 = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t beta2 = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t epsilon = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t step_size = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t inv_sqrt_bc2 = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t one_minus_beta1 = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t one_minus_beta2 = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t decay_factor = get_arg_val<uint32_t>(runtime_args_counter++);

    init_sfpu(cb_param_idx, cb_output_idx);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        // momentum_t calculation
        cb_wait_front(cb_exp_avg_idx, block_size);
        reconfig_data_format(cb_exp_avg_idx, cb_exp_avg_idx);
        copy_tile_to_dst_init_short(cb_exp_avg_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_exp_avg_idx, block_idx, block_idx);
        }
        cb_pop_front(cb_exp_avg_idx, block_size);
        // beta_1 * m_{t-1}
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_idx, beta1);
        }
        cb_wait_front(cb_grad_idx, block_size);
        reconfig_data_format(cb_grad_idx, cb_grad_idx);
        copy_tile_to_dst_init_short(cb_grad_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_idx, block_idx, block_size + block_idx);
        }
        // (1 - beta_1) * g_t
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_size + block_idx, one_minus_beta1);
        }
        // beta_1 * m_{t-1} + (1 - beta_1) * g_t
        add_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_two_blocks(cb_m_t, cb_exp_avg_out_idx, block_size);

        // variance_t calculation
        cb_wait_front(cb_exp_avg_sq_idx, block_size);
        reconfig_data_format(cb_exp_avg_sq_idx, cb_exp_avg_sq_idx);
        copy_tile_to_dst_init_short(cb_exp_avg_sq_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_exp_avg_sq_idx, block_idx, block_idx);
        }
        cb_pop_front(cb_exp_avg_sq_idx, block_size);
        // beta_2 * v_{t-1}
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_idx, beta2);
        }
        reconfig_data_format(cb_grad_idx, cb_grad_idx);
        copy_tile_to_dst_init_short(cb_grad_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_idx, block_idx, block_size + block_idx);
        }
        cb_pop_front(cb_grad_idx, block_size);

        square_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            square_tile(block_size + block_idx);
        }
        // (1 - beta_2) * g_t^2
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_size + block_idx, one_minus_beta2);
        }
        // beta_2 * v_t + (1 - beta_2) * g_t^2
        add_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_two_blocks(cb_v_t, cb_exp_avg_sq_out_idx, block_size);

        // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))
        cb_wait_front(cb_v_t, block_size);
        copy_tile_to_dst_init_short(cb_v_t);
        reconfig_data_format(cb_v_t, cb_v_t);
        tile_regs_acquire();
        for (uint32_t block_idx = 0, cb_tile_idx = 0; cb_tile_idx < block_size; block_idx += 2, ++cb_tile_idx) {
            copy_tile(cb_v_t, cb_tile_idx, block_idx);
        }
        cb_pop_front(cb_v_t, block_size);

#if AMSGRAD
        // TODO: I think this can be merged, check
        cb_wait_front(cb_max_exp_avg_sq_in_idx, block_size);
        copy_tile_to_dst_init_short(cb_max_exp_avg_sq_in_idx);
        reconfig_data_format(cb_max_exp_avg_sq_in_idx, cb_max_exp_avg_sq_in_idx);
        binary_max_tile_init();
        for (uint32_t block_idx = 0, cb_tile_idx = 0; cb_tile_idx < block_size; block_idx += 2, ++cb_tile_idx) {
            copy_tile(cb_max_exp_avg_sq_in_idx, cb_tile_idx, block_idx + 1);
            // This part used max_tile API previously that's why it's the every second block
            binary_max_tile(block_idx, block_idx + 1, block_idx);
        }
        cb_pop_front(cb_max_exp_avg_sq_in_idx, block_size);
        tile_regs_commit();
        // push every second block to max_exp_avg_sq_out
        cb_reserve_back(cb_max_exp_avg_sq_out_idx, block_size);
        cb_reserve_back(cb_max_exp_avg_sq_idx, block_size);
        tile_regs_wait();
        pack_reconfig_data_format(cb_max_exp_avg_sq_out_idx);
        for (uint32_t block_idx = 0; block_idx < twice_block_size; block_idx += 2) {
            pack_tile(block_idx, cb_max_exp_avg_sq_out_idx);
        }
        pack_reconfig_data_format(cb_max_exp_avg_sq_idx);
        for (uint32_t block_idx = 0; block_idx < twice_block_size; block_idx += 2) {
            pack_tile(block_idx, cb_max_exp_avg_sq_idx);
        }
        tile_regs_release();
        cb_push_back(cb_max_exp_avg_sq_out_idx, block_size);
        cb_push_back(cb_max_exp_avg_sq_idx, block_size);

        cb_wait_front(cb_max_exp_avg_sq_idx, block_size);
        copy_tile_to_dst_init_short(cb_max_exp_avg_sq_idx);
        reconfig_data_format(cb_max_exp_avg_sq_idx, cb_max_exp_avg_sq_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0, cb_tile_idx = 0; cb_tile_idx < block_size; block_idx += 2, ++cb_tile_idx) {
            copy_tile(cb_max_exp_avg_sq_idx, cb_tile_idx, block_idx);
        }
        cb_pop_front(cb_max_exp_avg_sq_idx, block_size);
#endif
        sqrt_tile_init();
        for (uint32_t block_idx = 0; block_idx < twice_block_size; block_idx += 2) {
            sqrt_tile(block_idx);
        }
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < twice_block_size; block_idx += 2) {
            mul_unary_tile(block_idx, inv_sqrt_bc2);
            add_unary_tile(block_idx, epsilon);
        }
        cb_wait_front(cb_m_t, block_size);
        copy_tile_to_dst_init_short(cb_m_t);
        reconfig_data_format(cb_m_t, cb_m_t);
        for (uint32_t block_idx = 0, cb_tile_idx = 0; cb_tile_idx < block_size; block_idx += 2, ++cb_tile_idx) {
            copy_tile(cb_m_t, cb_tile_idx, block_idx + 1);
        }
        cb_pop_front(cb_m_t, block_size);

        div_binary_tile_init();
        uint32_t reg_ite = 0;
        for (uint32_t block_idx = 0; block_idx < twice_block_size; block_idx += 2) {
            div_binary_tile(block_idx + 1, block_idx, reg_ite);
            reg_ite++;
        }
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_idx, step_size);
        }
        cb_wait_front(cb_param_idx, block_size);
        reconfig_data_format(cb_param_idx, cb_param_idx);
        copy_tile_to_dst_init_short(cb_param_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_param_idx, block_idx, block_size + block_idx);
        }
        // 0x3F800000 is hexadecimal encoding of 1 in fp32
        if (decay_factor != fp32_one) {
            binop_with_scalar_tile_init();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_unary_tile(block_size + block_idx, decay_factor);
            }
        }
        sub_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sub_binary_tile(block_size + block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_output_idx, block_size);
        cb_pop_front(cb_param_idx, block_size);
    }
}
}  // namespace NAMESPACE
