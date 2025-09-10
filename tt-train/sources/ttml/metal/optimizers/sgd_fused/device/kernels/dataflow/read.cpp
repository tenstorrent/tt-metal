// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

constexpr auto kParamInCbIndex = tt::CBIndex::c_0;
constexpr auto kGradCbIndex = tt::CBIndex::c_1;
constexpr auto kLrCbIndex = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

template <typename AddrGen>
inline void read_tiles(
    const uint32_t cb_idx,
    const AddrGen& addr_gen,
    const uint32_t start_tile_idx,
    const uint32_t block_size,
    const uint32_t tile_size_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    cb_reserve_back(cb_idx, block_size);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t k = 0; k < block_size; ++k) {
        noc_async_read_tile(start_tile_idx + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_size_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t param_in_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t lr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    generate_tile_with_float32_value(kLrCbIndex, lr);

    const uint32_t tile_size_bytes = get_tile_size(kParamInCbIndex);

    constexpr auto param_in_args = TensorAccessorArgs<2U>();
    constexpr auto grad_args = TensorAccessorArgs<param_in_args.next_compile_time_args_offset()>();

    const auto param_in_addr_gen = TensorAccessor(param_in_args, param_in_addr, tile_size_bytes);
    const auto grad_addr_gen = TensorAccessor(grad_args, grad_addr, tile_size_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;

            read_tiles(kParamInCbIndex, param_in_addr_gen, row_tile_idx, block_size, tile_size_bytes);
            read_tiles(kGradCbIndex, grad_addr_gen, row_tile_idx, block_size, tile_size_bytes);
            noc_async_read_barrier();
            cb_push_back(kParamInCbIndex, block_size);
            cb_push_back(kGradCbIndex, block_size);
        }
    }
}
