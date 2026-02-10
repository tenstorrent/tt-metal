// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

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

// AdamW update:
//   m_t = β₁ * m_{t-1} + (1 - β₁) * g          (momentum)
//   v_t = β₂ * v_{t-1} + (1 - β₂) * g²         (variance)
//   θ_t = θ_{t-1} - step_size * m_t / (√v̂_t + ε)
void kernel_main() {
    // multiple kernels use this, can be moved to compute_utils.hpp
    constexpr uint32_t fp32_one = 0x3F800000U;  // hexadecimal encoding of 1.0f in uint32_t

    // Runtime args (per-core, set via SetRuntimeArgs)
    uint32_t args_idx = 0;
    uint32_t beta1 = get_arg_val<uint32_t>(args_idx++);
    uint32_t beta2 = get_arg_val<uint32_t>(args_idx++);
    uint32_t epsilon = get_arg_val<uint32_t>(args_idx++);
    uint32_t step_size = get_arg_val<uint32_t>(args_idx++);
    uint32_t inv_sqrt_bc2 = get_arg_val<uint32_t>(args_idx++);
    uint32_t one_minus_beta1 = get_arg_val<uint32_t>(args_idx++);
    uint32_t one_minus_beta2 = get_arg_val<uint32_t>(args_idx++);
    uint32_t decay_factor = get_arg_val<uint32_t>(args_idx++);
    [[maybe_unused]] uint32_t seed = get_arg_val<uint32_t>(args_idx++);

    init_sfpu(cb_param_idx, cb_output_idx);
#if STOCH_ROUND
    init_prng_seed(seed);
#endif

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        // momentum_t calculation
        cb_wait_front(cb_exp_avg_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_param_idx, cb_exp_avg_idx);
        binop_with_scalar_tile_init();
        tile_regs_acquire();
        // beta_1 * m_{t-1}
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_exp_avg_idx, block_idx, block_idx);
            mul_unary_tile(block_idx, beta1);
        }
        cb_pop_front(cb_exp_avg_idx, block_size);
        cb_wait_front(cb_grad_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_exp_avg_idx, cb_grad_idx);
        binop_with_scalar_tile_init();
        // beta_1 * m_{t-1} + (1 - beta_1) * g
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_idx, block_idx, block_size + block_idx);
            mul_unary_tile(block_size + block_idx, one_minus_beta1);
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_two_blocks(cb_m_t, cb_exp_avg_out_idx, block_size);

        // variance_t calculation
        cb_wait_front(cb_exp_avg_sq_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_grad_idx, cb_exp_avg_sq_idx);
        binop_with_scalar_tile_init();
        tile_regs_acquire();
        // beta_2 * v_{t-1}
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_exp_avg_sq_idx, block_idx, block_idx);
            mul_unary_tile(block_idx, beta2);
        }
        cb_pop_front(cb_exp_avg_sq_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_exp_avg_sq_idx, cb_grad_idx);
        square_tile_init();
        // binop_with_scalar_tile_init(); this and square_tile_init() are the same
        // add_binary_tile_init(); same with this init
        // beta_2 * v_t + (1 - beta_2) * g_t^2
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_idx, block_idx, block_size + block_idx);
            square_tile(block_size + block_idx);
            mul_unary_tile(block_size + block_idx, one_minus_beta2);
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        cb_pop_front(cb_grad_idx, block_size);
        tile_regs_commit();
        pack_and_push_two_blocks(cb_v_t, cb_exp_avg_sq_out_idx, block_size);

        cb_wait_front(cb_v_t, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_grad_idx, cb_v_t);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_v_t, block_idx, block_idx);
        }
        cb_pop_front(cb_v_t, block_size);

#if AMSGRAD
        // AMSGrad: use max of past squared gradients
        cb_wait_front(cb_max_exp_avg_sq_in_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_v_t, cb_max_exp_avg_sq_in_idx);
        binary_max_tile_init();
        // v_t = max(v_max_t, v_t)
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_max_exp_avg_sq_in_idx, block_idx, block_size + block_idx);
            binary_max_tile(block_idx, block_size + block_idx, block_idx);
        }
        cb_pop_front(cb_max_exp_avg_sq_in_idx, block_size);
        tile_regs_commit();
        pack_and_push_two_blocks(cb_max_exp_avg_sq_out_idx, cb_max_exp_avg_sq_idx, block_size);

        cb_wait_front(cb_max_exp_avg_sq_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_max_exp_avg_sq_in_idx, cb_max_exp_avg_sq_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_max_exp_avg_sq_idx, block_idx, block_idx);
        }
        cb_pop_front(cb_max_exp_avg_sq_idx, block_size);
#endif
        sqrt_tile_init();  // sets extra constants
        // binop_with_scalar_tile_init();
        // sqrt(v_hat_t) + epsilon = sqrt(v_t) * inv_sqrt_bc2 + epsilon)
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sqrt_tile(block_idx);
            mul_unary_tile(block_idx, inv_sqrt_bc2);
            add_unary_tile(block_idx, epsilon);
        }
        cb_wait_front(cb_m_t, block_size);
        copy_tile_init(cb_m_t);
        div_binary_tile_init();
        // m_t / (sqrt(v_hat_t) + epsilon)
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_m_t, block_idx, block_size + block_idx);
            div_binary_tile(block_size + block_idx, block_idx, block_idx);
        }
        cb_pop_front(cb_m_t, block_size);

        cb_wait_front(cb_param_idx, block_size);
        copy_tile_to_dst_init_short_with_dt(cb_m_t, cb_param_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_param_idx, block_idx, block_size + block_idx);
        }
        binop_with_scalar_tile_init();
        if (decay_factor != fp32_one) {
            // theta_t = decay_factory * theta_{t - 1} = (1 - weight_decay * learning_rate) * theta_{t - 1}
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_unary_tile(block_size + block_idx, decay_factor);
            }
        }
        // theta_t = theta_t - step_size * update, where step_size = learning_rate / bias_correction1
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_idx, step_size);
            sub_binary_tile(block_size + block_idx, block_idx, block_idx);
        }
#if STOCH_ROUND
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            stochastic_round_tile(block_idx);
        }
#endif
        tile_regs_commit();
        pack_and_push_block(cb_output_idx, block_size);
        cb_pop_front(cb_param_idx, block_size);
    }
}
