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
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dgate_idx);
    constexpr auto dpacked_args = TensorAccessorArgs<2>();
    const auto dpacked_gen = TensorAccessor(dpacked_args, dpacked_addr);

    constexpr uint32_t packed_row_tiles = 2U * Wt;
    constexpr uint32_t blocks_per_row = Wt / block_size;

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
        const uint32_t dgate_start = (b / blocks_per_row) * packed_row_tiles + (b % blocks_per_row) * block_size;
        // Disjoint tiles of the same dpacked buffer.
        write_tiles_by_row(cb_dgate_idx, dpacked_gen, dgate_start, block_size, tile_bytes, block_size);
        write_tiles_by_row(cb_dup_idx, dpacked_gen, dgate_start + Wt, block_size, tile_bytes, block_size);
    }
}
