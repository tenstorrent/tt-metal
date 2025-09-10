// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr auto kParamInCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kLrCbIndex = tt::CBIndex::c_2;
constexpr auto kUpdateCbIndex = tt::CBIndex::c_3;

constexpr auto kOutputCbIndex = tt::CBIndex::c_16;

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

void MAIN {
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(kGradCbIndex, block_size);
            cb_wait_front(kLrCbIndex, 1U);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                mul_tiles_init(kGradCbIndex, kLrCbIndex);
                mul_tiles(kGradCbIndex, kLrCbIndex, block_idx, 0, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(kUpdateCbIndex, block_size);

            cb_wait_front(kParamInCbIndex, block_size);
            tile_regs_acquire();
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                sub_tiles_init(kParamInCbIndex, kUpdateCbIndex);
                sub_tiles(kParamInCbIndex, kUpdateCbIndex, block_idx, block_idx, block_idx);
            }
            tile_regs_commit();
            pack_and_push_block(kOutputCbIndex, block_size);

            cb_pop_front(kParamInCbIndex, block_size);
            cb_pop_front(kGradCbIndex, block_size);
        }
    }
    cb_pop_front(kUpdateCbIndex, 1U);
}
}  // namespace NAMESPACE
