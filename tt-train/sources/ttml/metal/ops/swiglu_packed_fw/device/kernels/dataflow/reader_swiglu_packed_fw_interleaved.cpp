// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // output width in tiles (one branch)

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t packed_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_gate_idx);
    constexpr auto packed_args = TensorAccessorArgs<2>();
    const auto packed_gen = TensorAccessor(packed_args, packed_addr);

    // Packed row = 2*Wt tiles: gate [0, Wt), up [Wt, 2*Wt).
    constexpr uint32_t packed_row_tiles = 2U * Wt;

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t gate_row_start = r * packed_row_tiles;
        const uint32_t up_row_start = gate_row_start + Wt;
        for (uint32_t c = 0; c < Wt; c += block_size) {
            const uint32_t current_block_size = std::min(block_size, Wt - c);

            read_tiles_by_row<false>(
                cb_gate_idx, packed_gen, gate_row_start + c, current_block_size, tile_bytes, block_size);
            read_tiles_by_row<false>(
                cb_up_idx, packed_gen, up_row_start + c, current_block_size, tile_bytes, block_size);
            noc_async_read_barrier();
            cb_push_back(cb_gate_idx, block_size);
            cb_push_back(cb_up_idx, block_size);
        }
    }
}
