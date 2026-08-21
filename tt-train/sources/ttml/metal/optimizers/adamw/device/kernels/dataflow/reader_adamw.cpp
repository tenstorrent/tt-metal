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
constexpr auto cb_scalars_idx = tt::CBIndex::c_5;

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
#if SCALARS_FROM_TENSOR
    const uint32_t step_size_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t inv_sqrt_bc2_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t decay_factor_addr = get_arg_val<uint32_t>(runtime_args_counter++);
#endif

    // Tile size in bytes for parameters and moving averages (can be bf16 or fp32)
    const uint32_t tile_size_bytes = get_tile_size(cb_param_idx);
    // Gradient is always bf16
    const uint32_t grad_tile_size_bytes = get_tile_size(cb_grad_idx);

    constexpr auto param_args = TensorAccessorArgs<1U>();
    constexpr auto grad_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();

    const auto param_addr_gen = TensorAccessor(param_args, param_addr);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr);
    const auto exp_avg_addr_gen = TensorAccessor(exp_avg_args, exp_avg_addr);
    const auto exp_avg_sq_addr_gen = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr);
    const auto max_exp_avg_sq_addr_gen = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr);

#if SCALARS_FROM_TENSOR
    constexpr auto step_size_args = TensorAccessorArgs<max_exp_avg_sq_args.next_compile_time_args_offset()>();
    constexpr auto inv_sqrt_bc2_args = TensorAccessorArgs<step_size_args.next_compile_time_args_offset()>();
    constexpr auto decay_factor_args = TensorAccessorArgs<inv_sqrt_bc2_args.next_compile_time_args_offset()>();

    // Each scalar is the single f32 at element (0, 0) of its padded tile, i.e. at
    // byte offset 0 of DRAM page 0 -- a sized read of just that element replaces a
    // full tile read per scalar. Slot stride must match the CB page size configured
    // by the host (get_tile_size() would report the format-derived 4 KB tile size).
    constexpr uint32_t scalar_slot_bytes = SCALAR_SLOT_BYTES;
    cb_reserve_back(cb_scalars_idx, 3U);
    const uint32_t scalars_l1_addr = get_write_ptr(cb_scalars_idx);
    noc_async_read(TensorAccessor(step_size_args, step_size_addr).get_noc_addr(0), scalars_l1_addr, sizeof(float));
    noc_async_read(
        TensorAccessor(inv_sqrt_bc2_args, inv_sqrt_bc2_addr).get_noc_addr(0),
        scalars_l1_addr + scalar_slot_bytes,
        sizeof(float));
    noc_async_read(
        TensorAccessor(decay_factor_args, decay_factor_addr).get_noc_addr(0),
        scalars_l1_addr + 2U * scalar_slot_bytes,
        sizeof(float));
    // No barrier here: the scalar reads ride along with the first block's tile reads
    // instead of stalling the reader for a full DRAM round trip up front. The push
    // happens after the first barrier below.
    bool scalars_pushed = false;
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
#if SCALARS_FROM_TENSOR
        if (!scalars_pushed) {
            cb_push_back(cb_scalars_idx, 3U);
            scalars_pushed = true;
        }
#endif
        cb_push_back(cb_param_idx, block_size);
        cb_push_back(cb_grad_idx, block_size);
        cb_push_back(cb_exp_avg_idx, block_size);
        cb_push_back(cb_exp_avg_sq_idx, block_size);
#if AMSGRAD
        cb_push_back(cb_max_exp_avg_sq_in_idx, block_size);
#endif
    }

#if SCALARS_FROM_TENSOR
    // A core with no tiles never reaches the loop's barrier; compute still waits on
    // the scalars CB, so flush and push here.
    if (!scalars_pushed) {
        noc_async_read_barrier();
        cb_push_back(cb_scalars_idx, 3U);
    }
#endif
}
