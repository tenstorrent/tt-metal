// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr auto cb_param_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_exp_avg_idx = tt::CBIndex::c_2;
constexpr auto cb_exp_avg_sq_idx = tt::CBIndex::c_3;
constexpr auto cb_max_exp_avg_sq_in_idx = tt::CBIndex::c_4;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t param_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t max_exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    // Tile size in bytes for parameters and moving averages (can be bf16 or fp32)
    const uint32_t tile_size_bytes = get_tile_size(cb_param_idx);
    // Gradient is always bf16
    const uint32_t grad_tile_size_bytes = get_tile_size(cb_grad_idx);

    constexpr auto param_args = TensorAccessorArgs<1U>();
    constexpr auto grad_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();

    const auto param_addr_gen = TensorAccessor(param_args, param_addr, tile_size_bytes);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr, grad_tile_size_bytes);
    const auto exp_avg_addr_gen = TensorAccessor(exp_avg_args, exp_avg_addr, tile_size_bytes);
    const auto exp_avg_sq_addr_gen = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr, tile_size_bytes);
    const auto max_exp_avg_sq_addr_gen = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr, tile_size_bytes);

    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        read_tiles_by_row</* UseBarrier = */ false>(
            cb_param_idx, param_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
        read_tiles_by_row</* UseBarrier = */ false>(
            cb_grad_idx, grad_addr_gen, tile_idx, current_block_size, grad_tile_size_bytes, block_size);
        read_tiles_by_row</* UseBarrier = */ false>(
            cb_exp_avg_idx, exp_avg_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
        read_tiles_by_row</* UseBarrier = */ false>(
            cb_exp_avg_sq_idx, exp_avg_sq_addr_gen, tile_idx, current_block_size, tile_size_bytes, block_size);
#if AMSGRAD
        read_tiles_by_row</* UseBarrier = */ false>(
            cb_max_exp_avg_sq_in_idx,
            max_exp_avg_sq_addr_gen,
            tile_idx,
            current_block_size,
            tile_size_bytes,
            block_size);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_param_idx, block_size);
        cb_push_back(cb_grad_idx, block_size);
        cb_push_back(cb_exp_avg_idx, block_size);
        cb_push_back(cb_exp_avg_sq_idx, block_size);
#if AMSGRAD
        cb_push_back(cb_max_exp_avg_sq_in_idx, block_size);
#endif
    }
}
