// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr auto cb_param_in_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;

constexpr auto cb_momentum_in_idx = tt::CBIndex::c_2;
constexpr auto cb_grad_wd_idx = tt::CBIndex::c_3;
constexpr auto cb_momentum_out_idx = tt::CBIndex::c_4;
constexpr auto cb_momentum_to_dram_idx = tt::CBIndex::c_5;
constexpr auto cb_update_idx = tt::CBIndex::c_6;

constexpr auto cb_output_idx = tt::CBIndex::c_16;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

inline void pack_and_push_two_cbs(uint32_t cb_output_1, uint32_t cb_output_2, uint32_t block_size) {
    cb_reserve_back(cb_output_1, block_size);
    cb_reserve_back(cb_output_2, block_size);
    tile_regs_wait();
    pack_reconfig_data_format(cb_output_1);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output_1);
    }
    pack_reconfig_data_format(cb_output_2);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output_2);
    }
    tile_regs_release();
    cb_push_back(cb_output_1, block_size);
    cb_push_back(cb_output_2, block_size);
};

constexpr uint32_t F32_ONE = 0x3f800000u;

inline bool is_f32_zero_bits(uint32_t u) noexcept {
    // -0 and +0 handling
    return (u << 1) == 0u;
}
inline bool is_f32_one_bits(uint32_t u) noexcept {
    return u == F32_ONE;
}

void MAIN {
    uint32_t runtime_args_counter = 0;
    uint32_t lr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t momentum = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t one_minus_dampening = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);

    init_sfpu(cb_grad_idx, cb_grad_idx);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        cb_wait_front(cb_grad_idx, block_size);
        cb_wait_front(cb_param_in_idx, block_size);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile_init(cb_grad_idx);
            const uint32_t grad_register = block_idx;
            copy_tile(cb_grad_idx, /* tile_idx */ block_idx, /* register_idx */ grad_register);

            // g_t <- g_t + weight_decay * theta_{t-1}
            if (!is_f32_zero_bits(weight_decay)) {
                copy_tile_init(cb_param_in_idx);
                const uint32_t theta_register = block_size + block_idx;
                copy_tile(cb_param_in_idx, /* tile_idx */ block_idx, /* register_idx */ theta_register);
                binop_with_scalar_tile_init();
                mul_unary_tile(theta_register, weight_decay);
                add_binary_tile_init();
                add_binary_tile(grad_register, theta_register, grad_register);
            }

            // g_t <- g_t * (1 - dampening)
            if (!is_f32_one_bits(one_minus_dampening)) {
                binop_with_scalar_tile_init();
                mul_unary_tile(grad_register, one_minus_dampening);
            }
        }
        tile_regs_commit();
        pack_and_push_block(cb_grad_wd_idx, block_size);
        cb_pop_front(cb_grad_idx, block_size);

        cb_wait_front(cb_grad_wd_idx, block_size);
#if USE_MOMENTUM
        cb_wait_front(cb_momentum_in_idx, block_size);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile_init(cb_momentum_in_idx);
            const uint32_t momentum_register = block_size + block_idx;
            copy_tile(cb_momentum_in_idx, /* tile_idx */ block_idx, /* register_idx */ momentum_register);

            // m_{t-1} * momentum
            binop_with_scalar_tile_init();
            mul_unary_tile(momentum_register, momentum);

            copy_tile_init(cb_grad_wd_idx);
            const uint32_t grad_wd_register = block_idx;
            copy_tile(cb_grad_wd_idx, /* tile_idx */ block_idx, /* register_idx */ grad_wd_register);
            // m_t <- g_t + m_{t-1} * momentum
            add_binary_tile_init();
            add_binary_tile(grad_wd_register, momentum_register, grad_wd_register);
        }
        tile_regs_commit();
        pack_and_push_two_cbs(cb_momentum_out_idx, cb_momentum_to_dram_idx, block_size);
        cb_pop_front(cb_momentum_in_idx, block_size);
#else
        tile_regs_acquire();
        copy_tile_init(cb_grad_wd_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile(cb_grad_wd_idx, /* tile_idx */ block_idx, /* register_idx */ block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_momentum_out_idx, block_size);
#endif
        // m_t in cb_momentum_out_idx
        // apply learning rate
        cb_wait_front(cb_momentum_out_idx, block_size);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            copy_tile_init(cb_momentum_out_idx);
            const uint32_t update_register = block_idx;
            copy_tile(cb_momentum_out_idx, /* tile_idx */ block_idx, /* register_idx */ update_register);
#if USE_NESTEROV
            // m_t * momentum
            binop_with_scalar_tile_init();
            mul_unary_tile(update_register, momentum);

            copy_tile_init(cb_grad_wd_idx);
            const uint32_t grad_register = block_size + block_idx;
            copy_tile(cb_grad_wd_idx, /* tile_idx */ block_idx, /* register_idx */ grad_register);
            // g_t + m_t{t} * momentum
            add_binary_tile_init();
            add_binary_tile(grad_register, update_register, update_register);
#endif
            // u_t <- learning_rate * (g_t + m_t * momentum) with nesterov OR
            // u_t <- learning_rate * m_t otherwise
            binop_with_scalar_tile_init();
            mul_unary_tile(update_register, lr);
        }
        tile_regs_commit();
        cb_pop_front(cb_grad_wd_idx, block_size);
        cb_pop_front(cb_momentum_out_idx, block_size);
        pack_and_push_block(cb_update_idx, block_size);

        cb_wait_front(cb_update_idx, block_size);
        tile_regs_acquire();
        reconfig_data_format(cb_param_in_idx, cb_update_idx);
        sub_tiles_init(cb_param_in_idx, cb_update_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sub_tiles(cb_param_in_idx, cb_update_idx, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        cb_pop_front(cb_param_in_idx, block_size);
        cb_pop_front(cb_update_idx, block_size);
        pack_and_push_block(cb_output_idx, block_size);
    }
}
}  // namespace NAMESPACE
