// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Reads, per tile row r of this core's range: the running and step logsumexp tiles (tile r of the
// (B, H, S, 32) tensors), then the accumulator and step-output tiles r * Wt .. r * Wt + Wt - 1 in
// blocks of block_size.
void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t accum_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t running_lse_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t step_output_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t step_intermediates_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_running_lse = tt::CBIndex::c_0;
    constexpr uint32_t cb_step_lse = tt::CBIndex::c_1;
    constexpr uint32_t cb_accum = tt::CBIndex::c_2;
    constexpr uint32_t cb_step_out = tt::CBIndex::c_3;
    constexpr uint32_t cb_col0_mask = tt::CBIndex::c_4;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr auto accum_args = TensorAccessorArgs<2>();
    constexpr auto running_lse_args = TensorAccessorArgs<accum_args.next_compile_time_args_offset()>();
    constexpr auto step_output_args = TensorAccessorArgs<running_lse_args.next_compile_time_args_offset()>();
    constexpr auto step_intermediates_args = TensorAccessorArgs<step_output_args.next_compile_time_args_offset()>();

    const auto accum_gen = TensorAccessor(accum_args, accum_address);
    const auto running_lse_gen = TensorAccessor(running_lse_args, running_lse_address);
    const auto step_output_gen = TensorAccessor(step_output_args, step_output_address);
    const auto step_intermediates_gen = TensorAccessor(step_intermediates_args, step_intermediates_address);

    const uint32_t accum_tile_bytes = get_tile_size(cb_accum);
    const uint32_t step_out_tile_bytes = get_tile_size(cb_step_out);

    // 1.0 in column 0, 0.0 elsewhere: keeps the new running logsumexp in the intermediates layout.
    generate_matmul_row_reduce_tile(cb_col0_mask);

    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        cb_reserve_back(cb_running_lse, onetile);
        cb_reserve_back(cb_step_lse, onetile);
        noc_async_read_page(row, running_lse_gen, get_write_ptr(cb_running_lse));
        noc_async_read_page(row, step_intermediates_gen, get_write_ptr(cb_step_lse));
        noc_async_read_barrier();
        cb_push_back(cb_running_lse, onetile);
        cb_push_back(cb_step_lse, onetile);

        for (uint32_t col = 0; col < Wt; col += block_size) {
            const uint32_t first_tile = row * Wt + col;
            read_tiles_by_row</* UseBarrier */ false>(
                cb_accum, accum_gen, first_tile, block_size, accum_tile_bytes, block_size);
            read_tiles_by_row</* UseBarrier */ false>(
                cb_step_out, step_output_gen, first_tile, block_size, step_out_tile_bytes, block_size);
            noc_async_read_barrier();
            cb_push_back(cb_accum, block_size);
            cb_push_back(cb_step_out, block_size);
        }
    }
}
