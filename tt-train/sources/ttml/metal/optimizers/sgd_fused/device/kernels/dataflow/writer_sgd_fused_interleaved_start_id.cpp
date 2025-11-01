// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_momentum_out_idx = tt::CBIndex::c_3;
constexpr auto cb_momentum_to_dram_idx = tt::CBIndex::c_4;

constexpr auto cb_param_out_idx = tt::CBIndex::c_16;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx,
    const AddrGen& addr_gen,
    uint32_t start_idx,
    uint32_t block_size,
    uint32_t current_block_size,
    uint32_t tile_size_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_write_addr = get_read_ptr(cb_idx);

    for (uint32_t k = 0; k < current_block_size; ++k) {
        noc_async_write_tile(start_idx + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_size_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t param_out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t momentum_out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_param_out_idx);
    constexpr auto param_out_args = TensorAccessorArgs<2U>();
    constexpr auto momentum_output_args = TensorAccessorArgs<param_out_args.next_compile_time_args_offset()>();

    const auto param_out_addr_generator = TensorAccessor(param_out_args, param_out_addr, tile_size_bytes);
#ifdef USE_MOMENTUM
    const auto momentum_out_addr_generator = TensorAccessor(momentum_output_args, momentum_out_addr, tile_size_bytes);
#endif
    uint32_t end_tile = start_tile + num_tiles_to_process;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

#if USE_MOMENTUM
        write_cb_block_to_dram(
            cb_momentum_to_dram_idx,
            momentum_out_addr_generator,
            tile_idx,
            block_size,
            current_block_size,
            tile_size_bytes);
#endif
        write_cb_block_to_dram(
            cb_param_out_idx, param_out_addr_generator, tile_idx, block_size, current_block_size, tile_size_bytes);
        noc_async_write_barrier();
#if USE_MOMENTUM
        cb_pop_front(cb_momentum_to_dram_idx, block_size);
#endif
        cb_pop_front(cb_param_out_idx, block_size);
    }
}
