// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr auto cb_param_in_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_momentum_in_idx = tt::CBIndex::c_2;

constexpr auto cb_param_wd_idx = tt::CBIndex::c_3;
constexpr auto cb_grad_wd_idx = tt::CBIndex::c_4;

constexpr auto cb_momentum_scaled_idx = tt::CBIndex::c_5;
constexpr auto cb_momentum_out_idx = tt::CBIndex::c_6;
constexpr auto cb_momentum_dram_idx = tt::CBIndex::c_7;

constexpr auto cb_grad_dampened_idx = tt::CBIndex::c_8;

constexpr auto cb_nesterov_momentum_idx = tt::CBIndex::c_9;
constexpr auto cb_nesterov_update_idx = tt::CBIndex::c_10;

constexpr auto cb_update_idx = tt::CBIndex::c_11;

constexpr auto cb_bcast_lr_idx = tt::CBIndex::c_12;
constexpr auto cb_bcast_momentum_idx = tt::CBIndex::c_13;
constexpr auto cb_bcast_one_minus_dampening_idx = tt::CBIndex::c_14;
constexpr auto cb_bcast_wd_idx = tt::CBIndex::c_15;

constexpr auto cb_output_idx = tt::CBIndex::c_16;

constexpr uint32_t num_tiles_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

void MAIN {
    uint32_t runtime_args_counter = 0;
    const bool use_weight_decay = get_arg_val<uint32_t>(runtime_args_counter++);
    const bool use_dampening = get_arg_val<uint32_t>(runtime_args_counter++);

    binary_op_init_common(cb_grad_idx, cb_bcast_lr_idx, cb_update_idx);

    cb_wait_front(cb_bcast_lr_idx, 1);
    cb_wait_front(cb_bcast_momentum_idx, 1);
    cb_wait_front(cb_bcast_one_minus_dampening_idx, 1);
    cb_wait_front(cb_bcast_wd_idx, 1);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; tile_idx += block_size) {
        uint32_t alias_grad_modified = cb_grad_idx;
        cb_wait_front(cb_param_in_idx, block_size);
        if (use_weight_decay) {
            // param * wd
            mul_tiles_bcast_scalar_init_short(cb_param_in_idx, cb_bcast_wd_idx);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_bcast_scalar(cb_param_in_idx, cb_bcast_wd_idx, block_idx, 0, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_param_wd_idx, block_size);

            // param * wd + grad
            cb_wait_front(cb_param_wd_idx, block_size);
            cb_wait_front(cb_grad_idx, block_size);
            add_tiles_init(cb_param_wd_idx, cb_grad_idx);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                add_tiles(cb_param_wd_idx, cb_grad_idx, block_idx, block_idx, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_grad_wd_idx, block_size);
            cb_pop_front(cb_param_wd_idx, block_size);
            cb_pop_front(cb_grad_idx, block_size);
            alias_grad_modified = cb_grad_wd_idx;
        }

        uint32_t alias_update_not_scaled = alias_grad_modified;
#if USE_MOMENTUM
        cb_wait_front(cb_momentum_in_idx, block_size);
        mul_tiles_bcast_scalar_init_short(cb_momentum_in_idx, cb_bcast_momentum_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(cb_momentum_in_idx, cb_bcast_momentum_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_momentum_scaled_idx, block_size);
        cb_pop_front(cb_momentum_in_idx, block_size);

        uint32_t alias_grad_dampened = alias_grad_modified;
        if (use_dampening) {
            cb_wait_front(alias_grad_modified, block_size);
            mul_tiles_bcast_scalar_init_short(alias_grad_modified, cb_bcast_one_minus_dampening_idx);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_bcast_scalar(alias_grad_modified, cb_bcast_one_minus_dampening_idx, block_idx, 0, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_grad_dampened_idx, block_size);
            cb_pop_front(alias_grad_modified, block_size);
            alias_grad_dampened = cb_grad_dampened_idx;
        }

        cb_wait_front(alias_grad_dampened, block_size);
        cb_wait_front(cb_momentum_scaled_idx, block_size);
        add_tiles_init(cb_momentum_scaled_idx, alias_grad_dampened);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_tiles(cb_momentum_scaled_idx, alias_grad_dampened, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_two_blocks(cb_momentum_out_idx, cb_momentum_dram_idx, block_size);
        cb_pop_front(cb_momentum_scaled_idx, block_size);
#if USE_NESTEROV
        cb_wait_front(cb_momentum_out_idx, block_size);
        mul_tiles_bcast_scalar_init_short(cb_momentum_out_idx, cb_bcast_momentum_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(cb_momentum_out_idx, cb_bcast_momentum_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_nesterov_momentum_idx, block_size);
        cb_pop_front(cb_momentum_out_idx, block_size);

        cb_wait_front(cb_nesterov_momentum_idx, block_size);
        add_tiles_init(cb_nesterov_momentum_idx, alias_grad_dampened);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            add_tiles(cb_nesterov_momentum_idx, alias_grad_dampened, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_nesterov_update_idx, block_size);
        cb_pop_front(cb_nesterov_momentum_idx, block_size);
        cb_pop_front(alias_grad_dampened, block_size);
        alias_update_not_scaled = cb_nesterov_update_idx;
#else
        cb_pop_front(alias_grad_dampened, block_size);
        alias_update_not_scaled = cb_momentum_out_idx;
#endif
#endif
        // grad * lr
        cb_wait_front(alias_update_not_scaled, block_size);
        mul_tiles_bcast_scalar_init_short(alias_update_not_scaled, cb_bcast_lr_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_tiles_bcast_scalar(alias_update_not_scaled, cb_bcast_lr_idx, block_idx, 0, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_update_idx, block_size);
        cb_pop_front(alias_update_not_scaled, block_size);

        // param - grad * lr
        cb_wait_front(cb_update_idx, block_size);
        sub_tiles_init(cb_param_in_idx, cb_update_idx);
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            sub_tiles(cb_param_in_idx, cb_update_idx, block_idx, block_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_output_idx, block_size);

        cb_pop_front(cb_param_in_idx, block_size);
        cb_pop_front(cb_update_idx, block_size);
    }
    cb_pop_front(cb_bcast_lr_idx, 1);
    cb_pop_front(cb_bcast_momentum_idx, 1);
    cb_pop_front(cb_bcast_one_minus_dampening_idx, 1);
    cb_pop_front(cb_bcast_wd_idx, 1);
}
}  // namespace NAMESPACE
