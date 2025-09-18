// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
constexpr auto cb_momentum_out_idx = tt::CBIndex::c_3;

constexpr auto cb_output_idx = tt::CBIndex::c_16;

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t momentum = get_compile_time_arg_val(3);

void MAIN {
    uint32_t runtime_args_counter = 0;
    uint32_t lr = get_arg_val<uint32_t>(runtime_args_counter++);

    init_sfpu(cb_grad_idx, cb_grad_idx);
    init_sfpu(cb_momentum_in_idx, cb_momentum_out_idx);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_grad_idx, block_size);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                copy_tile_init(cb_grad_idx);
                const uint32_t update_register = block_idx;
                copy_tile(cb_grad_idx, /* tile_idx */ block_idx, /* register_idx */ update_register);
                copy_tile_init(cb_momentum_in_idx);
                const uint32_t momentum_register = block_size + block_idx;
                copy_tile(cb_momentum_in_idx, /* tile_idx */ block_idx, /* register_idx */ momentum_register);

                binop_with_scalar_tile_init();
                mul_unary_tile(block_idx, lr);
                mul_unary_tile(momentum_register, momentum);

                add_binary_tile_init();
                add_binary_tile(update_register, momentum_register, update_register);
            }
            tile_regs_commit();
            pack_and_push_block(cb_momentum_out_idx, block_size);

            cb_pop_front(cb_grad_idx, block_size);

            cb_wait_front(cb_param_in_idx, block_size);
            cb_wait_front(cb_momentum_out_idx, block_size);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                sub_tiles_init(cb_param_in_idx, cb_momentum_out_idx);  // TODO: Check if here or outside loop
                sub_tiles(cb_param_in_idx, cb_momentum_out_idx, block_idx, block_idx, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(cb_output_idx, block_size);

            cb_pop_front(cb_param_in_idx, block_size);
        }
    }
}
}  // namespace NAMESPACE
