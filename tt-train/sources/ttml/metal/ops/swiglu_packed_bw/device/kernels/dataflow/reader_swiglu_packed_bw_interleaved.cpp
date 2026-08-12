// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;  // gate branch (silu'd), from packed[:, :I]
constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;    // up branch (plain),   from packed[:, I:]
constexpr uint32_t cb_dh_idx = tt::CBIndex::c_2;    // upstream grad dL/dh

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);  // half width in tiles (one branch)

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t packed_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dh_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_gate_idx);
    constexpr auto packed_args = TensorAccessorArgs<2>();
    constexpr auto dh_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    const auto packed_gen = TensorAccessor(packed_args, packed_addr);
    const auto dh_gen = TensorAccessor(dh_args, dh_addr);

    constexpr uint32_t packed_row_tiles = 2U * Wt;
    constexpr uint32_t blocks_per_row = Wt / block_size;

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
        const uint32_t gate_start = (b / blocks_per_row) * packed_row_tiles + (b % blocks_per_row) * block_size;

        read_tiles_by_row<false>(cb_gate_idx, packed_gen, gate_start, block_size, tile_bytes, block_size);
        read_tiles_by_row<false>(cb_up_idx, packed_gen, gate_start + Wt, block_size, tile_bytes, block_size);
        read_tiles_by_row<false>(cb_dh_idx, dh_gen, b * block_size, block_size, tile_bytes, block_size);
        noc_async_read_barrier();
        cb_push_back(cb_gate_idx, block_size);
        cb_push_back(cb_up_idx, block_size);
        cb_push_back(cb_dh_idx, block_size);
    }
}
