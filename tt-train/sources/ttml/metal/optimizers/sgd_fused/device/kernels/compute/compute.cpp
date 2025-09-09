// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {

constexpr auto kParamInCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kLrCbIndex = tt::CBIndex::c_2;

constexpr auto kOutputCbIndex = tt::CBIndex::c_16;

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

void MAIN() {
    uint32_t runtime_args_counter = 0;
    uint32_t lr = get_arg_val<uint32_t>(runtime_args_counter++);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(kParamInCbIndex, block_size);
            cb_wait_front(kGradCbIndex, block_size);
            cb_reserve_back(kOutputCbIndex, block_size);

            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                // binop_with_scalar_tile_init();
                // mul_unary_tile(grad_register, lr);

                sub_tiles_init(kParamInCbIndex, kGradCbIndex);
                sub_tiles(kParamInCbIndex, kGradCbIndex, block_idx, block_idx, block_idx);
            }
            tile_regs_commit();

            pack_and_push_block(kOutputCbIndex, block_size);

            cb_pop_front(kParamInCbIndex, block_size);
            cb_pop_front(kGradCbIndex, block_size);
        }
    }
}
}  // namespace NAMESPACE
