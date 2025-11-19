// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

namespace NAMESPACE {

constexpr auto cb_param_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_exp_avg_idx = tt::CBIndex::c_2;
constexpr auto cb_exp_avg_sq_idx = tt::CBIndex::c_3;

constexpr auto cb_m_t = tt::CBIndex::c_24;
constexpr auto cb_v_t = tt::CBIndex::c_25;
constexpr auto cb_output_idx = tt::CBIndex::c_16;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

void MAIN {
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

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        // momentum_t calculation
        cb_wait_front(cb_exp_avg_idx, block_size);
        reconfig_data_format_srca(cb_exp_avg_idx);
        copy_tile_init(cb_exp_avg_idx);
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
        reconfig_data_format_srca(cb_grad_idx);  // Check if can remove
        copy_tile_init(cb_grad_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_idx, block_size + block_idx);
        }
        // (1 - beta_1) * g_t
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_size + block_idx, one_minus_beta1);
        }
        add_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_m_t, block_size);

        // variance_t calculation
        cb_wait_front(cb_exp_avg_sq_idx, block_size);
        reconfig_data_format_srca(cb_exp_avg_sq_idx);
        copy_tile_init(cb_exp_avg_sq_idx);
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
        reconfig_data_format_srca(cb_grad_idx);  // Check if can remove
        copy_tile_init(cb_grad_idx);
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
        add_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_v_t, block_size);

        // theta_t = theta_{t-1} - step_size * (m_t / ((sqrt(v_t) * inv_sqrt_bc2) + epsilon))
        cb_wait_front(cb_v_t, block_size);
        copy_tile_init(cb_v_t);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_v_t, block_idx, block_size + block_idx);
        }
        cb_pop_front(cb_v_t, block_size);

        sqrt_tiles_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sqrt_tile(block_size + block_idx);
        }
        // TODO: Check if can be merged
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_size + block_idx, inv_sqrt_bc2);
        }
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_unary_tile(block_size + block_idx, epsilon);
        }
        cb_wait_front(cb_m_t, block_size);
        copy_tile_init(cb_m_t);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_m_t, block_idx, block_idx);
        }
        cb_pop_front(cb_m_t, block_size);
        // TODO: Can be put into seperate loop; Worth it?
        div_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            div_binary_tile(block_idx, block_size + block_idx, block_idx);
        }
        binop_with_scalar_tile_init();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_unary_tile(block_idx, step_size);
        }
        cb_wait_front(cb_param_idx, block_size);
        reconfig_data_format_srca(cb_param_idx);
        copy_tile_init(cb_param_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_param_idx, block_idx, block_size + block_idx);
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
