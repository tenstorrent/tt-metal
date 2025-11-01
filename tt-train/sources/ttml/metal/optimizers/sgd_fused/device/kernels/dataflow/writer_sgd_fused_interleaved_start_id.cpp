// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <debug/dprint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

constexpr auto cb_momentum_dram_idx = tt::CBIndex::c_7;
constexpr auto cb_output_idx = tt::CBIndex::c_16;

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
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t momentum_dram_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_tiles_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_size_bytes = get_tile_size(cb_output_idx);

    constexpr auto output_args = TensorAccessorArgs<1U>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr, tile_size_bytes);
#if USE_MOMENTUM
    constexpr auto momentum_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto momentum_dram_addr_generator = TensorAccessor(momentum_dram_args, momentum_dram_addr, tile_size_bytes);
#endif

    uint32_t end_tile = start_tile + num_tiles_to_process;
    uint32_t iteration = 0;
    for (uint32_t tile_idx = start_tile; tile_idx < end_tile; tile_idx += block_size) {
        uint32_t tiles_left = end_tile - tile_idx;
        uint32_t current_block_size = std::min(block_size, tiles_left);

        write_cb_block_to_dram(
            cb_output_idx, output_addr_generator, tile_idx, block_size, current_block_size, tile_size_bytes);
#if USE_MOMENTUM
        write_cb_block_to_dram(
            cb_momentum_dram_idx,
            momentum_dram_addr_generator,
            tile_idx,
            block_size,
            current_block_size,
            tile_size_bytes);
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_output_idx, block_size);
#if USE_MOMENTUM
        cb_pop_front(cb_momentum_dram_idx, block_size);
#endif
        iteration++;
    }
}
