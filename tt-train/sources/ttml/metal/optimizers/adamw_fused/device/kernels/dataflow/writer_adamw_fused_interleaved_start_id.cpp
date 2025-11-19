// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

constexpr auto cb_param_out_idx = tt::CBIndex::c_16;
constexpr auto cb_exp_avg_out_idx = tt::CBIndex::c_17;
constexpr auto cb_exp_avg_sq_out_idx = tt::CBIndex::c_18;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx,
    const AddrGen& addr_gen,
    uint32_t start_idx,
    uint32_t block_size,
    uint32_t current_block_size,
    uint32_t tile_size_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);

    for (uint32_t k = 0; k < current_block_size; ++k) {
        noc_async_write_tile(start_idx + k, addr_gen, l1_read_addr);
        l1_read_addr += tile_size_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t param_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t exp_avg_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t exp_avg_sq_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_out_idx);
    // TODO: Remember about tile size if changing datatype for momentum buffers
    // const uint32_t momentum_tile_size_bytes = get_tile_size(cb_exp_avg_idx);

    constexpr auto param_out_args = TensorAccessorArgs<1U>();
    constexpr auto exp_avg_out_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();
    constexpr auto exp_avg_sq_out_args = TensorAccessorArgs<exp_avg_out_args.next_compile_time_args_offset()>();

    const auto param_out_addr_gen = TensorAccessor(param_out_args, param_addr, tile_size_bytes);
    const auto exp_avg_out_addr_gen = TensorAccessor(exp_avg_out_args, exp_avg_addr, tile_size_bytes);
    const auto exp_avg_sq_out_addr_gen = TensorAccessor(exp_avg_sq_out_args, exp_avg_sq_addr, tile_size_bytes);

    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        write_cb_block_to_dram(
            cb_param_out_idx, param_out_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        write_cb_block_to_dram(
            cb_exp_avg_out_idx, exp_avg_out_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        write_cb_block_to_dram(
            cb_exp_avg_sq_out_idx, exp_avg_sq_out_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_param_out_idx, block_size);
        cb_pop_front(cb_exp_avg_out_idx, block_size);
        cb_pop_front(cb_exp_avg_sq_out_idx, block_size);
    }
}
