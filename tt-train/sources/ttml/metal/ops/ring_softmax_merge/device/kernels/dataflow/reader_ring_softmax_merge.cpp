// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per row tile: the running lse tile, the step's lse tile, then the running
// output row and the step's output row (Wt tiles each).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t lse_acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t step_lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t step_out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t rows = get_arg_val<uint32_t>(arg++);
    const uint32_t first_row = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto lse_acc_args = TensorAccessorArgs<1>();
    constexpr auto step_lse_args = TensorAccessorArgs<lse_acc_args.next_compile_time_args_offset()>();
    constexpr auto out_acc_args = TensorAccessorArgs<step_lse_args.next_compile_time_args_offset()>();
    constexpr auto step_out_args = TensorAccessorArgs<out_acc_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_lse_acc = tt::CBIndex::c_0;
    constexpr uint32_t cb_step_lse = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_acc = tt::CBIndex::c_2;
    constexpr uint32_t cb_step_out = tt::CBIndex::c_3;
    const uint32_t fp32_bytes = get_tile_size(cb_lse_acc);
    const uint32_t bf16_bytes = get_tile_size(cb_step_out);

    const auto lse_acc = TensorAccessor(lse_acc_args, lse_acc_addr, fp32_bytes);
    const auto step_lse = TensorAccessor(step_lse_args, step_lse_addr, fp32_bytes);
    const auto out_acc = TensorAccessor(out_acc_args, out_acc_addr, fp32_bytes);
    const auto step_out = TensorAccessor(step_out_args, step_out_addr, bf16_bytes);

    for (uint32_t r = first_row; r < first_row + rows; ++r) {
        read_tiles_by_row(cb_lse_acc, lse_acc, r, 1U, fp32_bytes, 1U);
        read_tiles_by_row(cb_step_lse, step_lse, r, 1U, fp32_bytes, 1U);
        read_tiles_by_row(cb_out_acc, out_acc, r * Wt, Wt, fp32_bytes, Wt);
        read_tiles_by_row(cb_step_out, step_out, r * Wt, Wt, bf16_bytes, Wt);
    }
}
