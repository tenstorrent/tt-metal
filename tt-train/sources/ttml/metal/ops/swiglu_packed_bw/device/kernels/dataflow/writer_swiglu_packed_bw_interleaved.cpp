// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_dgate_idx = tt::CBIndex::c_3;  // grad wrt gate branch -> dpacked[:, :I]
constexpr uint32_t cb_dup_idx = tt::CBIndex::c_4;    // grad wrt up branch   -> dpacked[:, I:]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // half width in tiles (one branch)

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t dpacked_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dgate_idx);
    constexpr auto dpacked_args = TensorAccessorArgs<2>();
    const auto dpacked_gen = TensorAccessor(dpacked_args, dpacked_addr);

    constexpr uint32_t packed_row_tiles = 2U * Wt;

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t dgate_row_start = r * packed_row_tiles;
        const uint32_t dup_row_start = dgate_row_start + Wt;
        for (uint32_t c = 0; c < Wt; c += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - c);
            // Disjoint tiles of the same dpacked buffer.
            write_tiles_by_row(
                cb_dgate_idx, dpacked_gen, dgate_row_start + c, current_block_size, tile_bytes, block_size);
            write_tiles_by_row(cb_dup_idx, dpacked_gen, dup_row_start + c, current_block_size, tile_bytes, block_size);
        }
    }
}
