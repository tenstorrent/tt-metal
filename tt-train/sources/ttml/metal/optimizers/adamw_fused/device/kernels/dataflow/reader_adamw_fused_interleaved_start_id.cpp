// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

constexpr auto cb_param_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_exp_avg_idx = tt::CBIndex::c_2;
constexpr auto cb_exp_avg_sq_idx = tt::CBIndex::c_3;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

template <typename AddrGen>
inline void read_tiles(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t start_tile_idx,
    const uint32_t block_size,
    const uint32_t current_block_size,
    const uint32_t tile_size_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    cb_reserve_back(cb_idx, block_size);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t k = 0; k < current_block_size; ++k) {
        noc_async_read_tile(start_tile_idx + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_size_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t param_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t max_exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_idx);
    // TODO: Remember about tile size if changing datatype for momentum buffers
    // const uint32_t momentum_tile_size_bytes = get_tile_size(cb_exp_avg_idx);

    constexpr auto param_args = TensorAccessorArgs<1U>();
    constexpr auto grad_args = TensorAccessorArgs<param_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_args = TensorAccessorArgs<exp_avg_args.next_compile_time_args_offset()>();
    constexpr auto max_exp_avg_sq_args = TensorAccessorArgs<exp_avg_sq_args.next_compile_time_args_offset()>();

    const auto param_addr_gen = TensorAccessor(param_args, param_addr, tile_size_bytes);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr, tile_size_bytes);
    const auto exp_avg_addr_gen = TensorAccessor(exp_avg_args, exp_avg_addr, tile_size_bytes);
    const auto exp_avg_sq_addr_gen = TensorAccessor(exp_avg_sq_args, exp_avg_sq_addr, tile_size_bytes);
    const auto max_exp_avg_sq_addr_gen = TensorAccessor(max_exp_avg_sq_args, max_exp_avg_sq_addr, tile_size_bytes);

    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        read_tiles(cb_param_idx, param_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        read_tiles(cb_grad_idx, grad_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        read_tiles(cb_exp_avg_idx, exp_avg_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        read_tiles(cb_exp_avg_sq_idx, exp_avg_sq_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
#if AMSGRAD
        read_tiles(
            cb_max_exp_avg_sq_idx, max_exp_avg_sq_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_param_idx, block_size);
        cb_push_back(cb_grad_idx, block_size);
        cb_push_back(cb_exp_avg_idx, block_size);
        cb_push_back(cb_exp_avg_sq_idx, block_size);
#if AMSGRAD
        cb_push_back(cb_max_exp_avg_sq_idx, block_size);
#endif
    }
}
