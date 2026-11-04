// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Writes, per tile row r of this core's range, the merged logsumexp tile back to tile r of the
// running logsumexp and the merged accumulator tiles back to r * Wt .. r * Wt + Wt - 1 (in place:
// the reader has already consumed those tiles of this row before compute produced them).
void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t accum_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t running_lse_address = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_new_lse = tt::CBIndex::c_5;
    constexpr uint32_t cb_out = tt::CBIndex::c_9;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr auto accum_args = TensorAccessorArgs<2>();
    constexpr auto running_lse_args = TensorAccessorArgs<accum_args.next_compile_time_args_offset()>();

    const auto accum_gen = TensorAccessor(accum_args, accum_address);
    const auto running_lse_gen = TensorAccessor(running_lse_args, running_lse_address);

    const uint32_t lse_tile_bytes = get_tile_size(cb_new_lse);
    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        write_tiles_by_row</* UseBarrier */ false>(cb_new_lse, running_lse_gen, row, onetile, lse_tile_bytes, onetile);
        for (uint32_t col = 0; col < Wt; col += block_size) {
            write_tiles_by_row</* UseBarrier */ false>(
                cb_out, accum_gen, row * Wt + col, block_size, out_tile_bytes, block_size);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, block_size);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_new_lse, onetile);
    }
}
