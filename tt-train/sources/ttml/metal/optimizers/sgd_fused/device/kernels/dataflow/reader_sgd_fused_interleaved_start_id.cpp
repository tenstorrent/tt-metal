// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

constexpr auto cb_param_in_idx = tt::CBIndex::c_0;
constexpr auto cb_grad_idx = tt::CBIndex::c_1;
constexpr auto cb_momentum_in_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

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
    const uint32_t param_in_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t momentum_in_addr = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_in_idx);

    constexpr auto param_in_args = TensorAccessorArgs<2U>();
    constexpr auto grad_args = TensorAccessorArgs<param_in_args.next_compile_time_args_offset()>();
    constexpr auto momentum_in_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();

    const auto param_in_addr_gen = TensorAccessor(param_in_args, param_in_addr, tile_size_bytes);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr, tile_size_bytes);
#if USE_MOMENTUM
    const auto momentum_in_addr_gen = TensorAccessor(momentum_in_args, momentum_in_addr, tile_size_bytes);
#endif

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            uint32_t current_block_size = (c + block_size <= Wt) ? block_size : (Wt - c);

            read_tiles(
                cb_param_in_idx, param_in_addr_gen, row_tile_idx, block_size, current_block_size, tile_size_bytes);
            read_tiles(cb_grad_idx, grad_addr_gen, row_tile_idx, block_size, current_block_size, tile_size_bytes);

#if USE_MOMENTUM
            read_tiles(
                cb_momentum_in_idx,
                momentum_in_addr_gen,
                row_tile_idx,
                block_size,
                current_block_size,
                tile_size_bytes);
#endif
            noc_async_read_barrier();
            cb_push_back(cb_param_in_idx, block_size);
            cb_push_back(cb_grad_idx, block_size);
#if USE_MOMENTUM
            cb_push_back(cb_momentum_in_idx, block_size);
#endif
        }
    }
}
