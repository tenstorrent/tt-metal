// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
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
constexpr auto cb_grad_dampened_idx = tt::CBIndex::c_3;
constexpr auto cb_grad_wd_idx = tt::CBIndex::c_4;
constexpr auto cb_momentum_out_idx = tt::CBIndex::c_5;
constexpr auto cb_momentum_to_dram_idx = tt::CBIndex::c_6;
constexpr auto cb_update_idx = tt::CBIndex::c_7;

constexpr auto cb_bcast_lr_idx = tt::CBIndex::c_8;
constexpr auto cb_bcast_momentum_idx = tt::CBIndex::c_9;
constexpr auto cb_bcast_dampening_idx = tt::CBIndex::c_10;
constexpr auto cb_bcast_weight_decay_idx = tt::CBIndex::c_11;

constexpr auto cb_param_wd_idx = tt::CBIndex::c_12;
constexpr auto cb_momentum_scaled_idx = tt::CBIndex::c_13;
constexpr auto cb_nesterov_momentum_scaled_idx = tt::CBIndex::c_14;
constexpr auto cb_nesterov_update_idx = tt::CBIndex::c_15;

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

    binary_op_init_common(cb_grad_idx, cb_bcast_lr_idx, cb_update_idx);
    binary_op_init_common(cb_param_in_idx, cb_update_idx, cb_output_idx);

    cb_wait_front(cb_bcast_lr_idx, 1);
    cb_wait_front(cb_bcast_momentum_idx, 1);
    cb_wait_front(cb_bcast_dampening_idx, 1);
    cb_wait_front(cb_bcast_weight_decay_idx, 1);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        uint32_t alias_grad_modified_idx = cb_grad_idx;
        cb_wait_front(cb_param_in_idx, block_size);

        if (!is_f32_zero_bits(weight_decay)) {
            cb_wait_front(alias_grad_modified_idx, block_size);
            tile_regs_acquire();

            mul_tiles_bcast_scalar_init_short(cb_param_in_idx, cb_bcast_weight_decay_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_bcast_scalar(cb_param_in_idx, cb_bcast_weight_decay_idx, block_idx, 0, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_param_wd_idx, block_size);

            cb_wait_front(cb_param_wd_idx, block_size);
            tile_regs_acquire();
            add_tiles_init(alias_grad_modified_idx, cb_param_wd_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                add_tiles(alias_grad_modified_idx, cb_param_wd_idx, block_idx, block_idx, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_grad_wd_idx, block_size);
            cb_pop_front(alias_grad_modified_idx, block_size);
            cb_pop_front(cb_param_wd_idx, block_size);
            alias_grad_modified_idx = cb_grad_wd_idx;
        }

        uint32_t alias_momentum_modified_idx = cb_momentum_in_idx;
#if USE_MOMENTUM
        // dampening only applied if momentum is used
        if (!is_f32_one_bits(one_minus_dampening)) {
            cb_wait_front(alias_grad_modified_idx, block_size);
            tile_regs_acquire();
            mul_tiles_bcast_scalar_init_short(alias_grad_modified_idx, cb_bcast_dampening_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_bcast_scalar(alias_grad_modified_idx, cb_bcast_dampening_idx, block_idx, 0, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_grad_dampened_idx, block_size);
            cb_pop_front(alias_grad_modified_idx, block_size);
            alias_grad_modified_idx = cb_grad_dampened_idx;
        }

        cb_wait_front(alias_momentum_modified_idx, block_size);
        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short(alias_momentum_modified_idx, cb_bcast_momentum_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(alias_momentum_modified_idx, cb_bcast_momentum_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_momentum_scaled_idx, block_size);
        cb_pop_front(alias_momentum_modified_idx, block_size);

        cb_wait_front(cb_momentum_scaled_idx, block_size);
        cb_wait_front(alias_grad_modified_idx, block_size);
        tile_regs_acquire();
        add_tiles_init(alias_grad_modified_idx, cb_momentum_scaled_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_tiles(alias_grad_modified_idx, cb_momentum_scaled_idx, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_two_cbs(cb_momentum_out_idx, cb_momentum_to_dram_idx, block_size);
        cb_pop_front(cb_momentum_scaled_idx, block_size);
        alias_momentum_modified_idx = cb_momentum_out_idx;
#if USE_NESTEROV
        cb_wait_front(alias_momentum_modified_idx, block_size);
        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short(alias_momentum_modified_idx, cb_bcast_momentum_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(alias_momentum_modified_idx, cb_bcast_momentum_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_nesterov_momentum_scaled_idx, block_size);
        cb_pop_front(alias_momentum_modified_idx, block_size);

        cb_wait_front(cb_nesterov_momentum_scaled_idx, block_size);
        tile_regs_acquire();
        add_tiles_init(cb_nesterov_momentum_scaled_idx, alias_grad_modified_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_tiles(cb_nesterov_momentum_scaled_idx, alias_grad_modified_idx, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        cb_pop_front(cb_nesterov_momentum_scaled_idx, block_size);
        cb_pop_front(alias_grad_modified_idx, block_size);
        pack_and_push_block(cb_nesterov_update_idx, block_size);
        alias_momentum_modified_idx = cb_nesterov_update_idx;
#else
        cb_pop_front(alias_grad_modified_idx, block_size);
#endif
#else
        alias_momentum_modified_idx = alias_grad_modified_idx;
#endif

        cb_wait_front(alias_momentum_modified_idx, block_size);
        tile_regs_acquire();
        mul_tiles_bcast_scalar_init_short(alias_momentum_modified_idx, cb_bcast_lr_idx);
        reconfig_data_format(cb_momentum_out_idx, cb_bcast_lr_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(alias_momentum_modified_idx, cb_bcast_lr_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        cb_pop_front(alias_momentum_modified_idx, block_size);
        pack_and_push_block(cb_update_idx, block_size);

        cb_wait_front(cb_update_idx, block_size);
        tile_regs_acquire();
        sub_tiles_init(cb_param_in_idx, cb_update_idx);
        reconfig_data_format(cb_param_in_idx, cb_update_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sub_tiles(cb_param_in_idx, cb_update_idx, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        cb_pop_front(cb_param_in_idx, block_size);
        cb_pop_front(cb_update_idx, block_size);
        pack_and_push_block(cb_output_idx, block_size);
    }
    cb_pop_front(cb_bcast_lr_idx, 1);
    cb_pop_front(cb_bcast_momentum_idx, 1);
    cb_pop_front(cb_bcast_dampening_idx, 1);
    cb_pop_front(cb_bcast_weight_decay_idx, 1);
}
}  // namespace NAMESPACE
