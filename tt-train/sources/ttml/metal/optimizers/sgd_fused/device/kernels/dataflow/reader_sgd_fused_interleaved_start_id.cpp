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

constexpr auto cb_bcast_lr_idx = tt::CBIndex::c_12;
constexpr auto cb_bcast_momentum_idx = tt::CBIndex::c_13;
constexpr auto cb_bcast_dampening_idx = tt::CBIndex::c_14;
constexpr auto cb_bcast_wd_idx = tt::CBIndex::c_15;
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
    const uint32_t param_in_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t momentum_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t packed_lr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t packed_momentum = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t packed_dampening = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t packed_wd = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_in_idx);

    constexpr auto param_in_args = TensorAccessorArgs<1U>();
    constexpr auto grad_args = TensorAccessorArgs<param_in_args.next_compile_time_args_offset()>();

    const auto param_in_addr_gen = TensorAccessor(param_in_args, param_in_addr, tile_size_bytes);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr, tile_size_bytes);
#if USE_MOMENTUM
    constexpr auto momentum_in_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    const auto momentum_in_addr_gen = TensorAccessor(momentum_in_args, momentum_addr, tile_size_bytes);
#endif

    generate_bcast_scalar_bfloat16(cb_bcast_lr_idx, packed_lr);
    generate_bcast_scalar_bfloat16(cb_bcast_momentum_idx, packed_momentum);
    generate_bcast_scalar_bfloat16(cb_bcast_dampening_idx, packed_dampening);
    generate_bcast_scalar_bfloat16(cb_bcast_wd_idx, packed_wd);

    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        read_tiles(cb_param_in_idx, param_in_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
        read_tiles(cb_grad_idx, grad_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
#if USE_MOMENTUM
        read_tiles(cb_momentum_in_idx, momentum_in_addr_gen, tile_idx, block_size, current_block_size, tile_size_bytes);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_param_in_idx, block_size);
        cb_push_back(cb_grad_idx, block_size);
#if USE_MOMENTUM
        cb_push_back(cb_momentum_in_idx, block_size);
#endif
    }
}
