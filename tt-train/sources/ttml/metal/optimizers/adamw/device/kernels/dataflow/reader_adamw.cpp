// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
constexpr auto cb_bias_correction_idx = tt::CBIndex::c_5;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t param_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t max_exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t beta1_pow_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t beta2_pow_addr = get_arg_val<uint32_t>(runtime_args_counter++);

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
    constexpr auto beta1_pow_args = TensorAccessorArgs<max_exp_avg_sq_args.next_compile_time_args_offset()>();
    constexpr auto beta2_pow_args = TensorAccessorArgs<beta1_pow_args.next_compile_time_args_offset()>();

    const auto param_addr_gen = TensorAccessor(param_args, param_addr);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr);
    const auto exp_avg_addr_gen = TensorAccessor(exp_avg_args, exp_avg_addr);
    const auto exp_avg_sq_addr_gen = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr);
    const auto max_exp_avg_sq_addr_gen = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr);

#if BIAS_CORRECTION_TENSORS
    // beta1^t and beta2^t arrive as single-element tensors. Fetch both once, before
    // the main loop: they do not vary per tile, and the compute kernel reads them
    // out of L1 to derive step_size and 1 / bias_correction2.
    //
    // Only the leading float of each tensor carries anything, and in TILE layout
    // element (0, 0) sits at byte offset 0, so this reads 4 bytes rather than
    // pulling a whole 4 KB tile per scalar onto every core. The two slots are
    // BIAS_SCALAR_STRIDE_BYTES apart to keep both NOC destinations aligned.
    {
        const auto beta1_pow_addr_gen = TensorAccessor(beta1_pow_args, beta1_pow_addr);
        const auto beta2_pow_addr_gen = TensorAccessor(beta2_pow_args, beta2_pow_addr);

        cb_reserve_back(cb_bias_correction_idx, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_bias_correction_idx);
        noc_async_read(get_noc_addr(0, beta1_pow_addr_gen), l1_write_addr, sizeof(float));
        noc_async_read(
            get_noc_addr(0, beta2_pow_addr_gen), l1_write_addr + BIAS_SCALAR_STRIDE_BYTES, sizeof(float));
        noc_async_read_barrier();
        cb_push_back(cb_bias_correction_idx, 1);
    }
#endif

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
