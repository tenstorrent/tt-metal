// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_linear1_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_gate_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_dL_dprod_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t linear1_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gate_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_dprod_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_linear1_idx);
    constexpr auto linear1_args = TensorAccessorArgs<2>();
    constexpr auto gate_args = TensorAccessorArgs<linear1_args.next_compile_time_args_offset()>();
    constexpr auto dL_dprod_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();
    const auto linear1_gen = TensorAccessor(linear1_args, linear1_addr, tile_bytes);
    const auto gate_gen = TensorAccessor(gate_args, gate_addr, tile_bytes);
    const auto dL_dprod_gen = TensorAccessor(dL_dprod_args, dL_dprod_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        for (uint32_t c = 0; c < Wt; c += block_size) {
            uint32_t row_tile_idx = (r * Wt) + c;
            uint32_t current_block_size = (c + block_size <= Wt) ? block_size : (Wt - c);

            read_tiles_by_row<false>(
                cb_linear1_idx, linear1_gen, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row<false>(cb_gate_idx, gate_gen, row_tile_idx, current_block_size, tile_bytes, block_size);
            read_tiles_by_row(cb_dL_dprod_idx, dL_dprod_gen, row_tile_idx, current_block_size, tile_bytes, block_size);
            cb_push_back(cb_linear1_idx, block_size);
            cb_push_back(cb_gate_idx, block_size);
        }
    }
}
