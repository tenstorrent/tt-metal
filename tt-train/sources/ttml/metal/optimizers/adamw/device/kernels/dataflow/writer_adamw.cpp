// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_param_out_idx = tt::CBIndex::c_16;
constexpr auto cb_exp_avg_out_idx = tt::CBIndex::c_17;
constexpr auto cb_exp_avg_sq_out_idx = tt::CBIndex::c_18;
constexpr auto cb_max_exp_avg_sq_out_idx = tt::CBIndex::c_19;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t param_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t exp_avg_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t max_exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_out_idx);

    constexpr auto param_out_args = TensorAccessorArgs<1U>();
    constexpr auto exp_avg_out_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_out_args = TensorAccessorArgs<exp_avg_out_args.next_compile_time_args_offset()>();
    constexpr auto max_exp_avg_sq_out_args = TensorAccessorArgs<exp_avg_sq_out_args.next_compile_time_args_offset()>();

    const auto param_out_addr_gen = TensorAccessor(param_out_args, param_addr, tile_size_bytes);
    const auto exp_avg_out_addr_gen = TensorAccessor(exp_avg_out_args, exp_avg_addr, tile_size_bytes);
    const auto exp_avg_sq_out_addr_gen = TensorAccessor(exp_avg_sq_out_args, exp_avg_sq_addr, tile_size_bytes);
    const auto max_exp_avg_sq_out_addr_gen =
        TensorAccessor(max_exp_avg_sq_out_args, max_exp_avg_sq_addr, tile_size_bytes);

    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        write_tiles_by_row</* UseBarrier = */ false>(
            cb_param_out_idx, param_out_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
        write_tiles_by_row</* UseBarrier = */ false>(
            cb_exp_avg_out_idx, exp_avg_out_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
        write_tiles_by_row</* UseBarrier = */ false>(
            cb_exp_avg_sq_out_idx, exp_avg_sq_out_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
#if AMSGRAD
        write_tiles_by_row</* UseBarrier = */ false>(
            cb_max_exp_avg_sq_out_idx,
            max_exp_avg_sq_out_addr_gen,
            tile_idx,
            current_block_size,
            tile_size_bytes,
            block_size);
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_param_out_idx, block_size);
        cb_pop_front(cb_exp_avg_out_idx, block_size);
        cb_pop_front(cb_exp_avg_sq_out_idx, block_size);
#if AMSGRAD
        cb_pop_front(cb_max_exp_avg_sq_out_idx, block_size);
#endif
    }
}
