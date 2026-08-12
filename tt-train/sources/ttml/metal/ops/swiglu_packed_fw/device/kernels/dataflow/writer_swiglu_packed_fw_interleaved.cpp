// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_out_idx = tt::CBIndex::c_2;

constexpr uint32_t block_size = get_compile_time_arg_val(0);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out_gen = TensorAccessor(out_args, out_addr);

    // The output is [.., R, I] = Wt tiles per row, so its tiles are exactly the blocks laid end to
    // end: block b covers output tiles [b*block_size, (b+1)*block_size).
    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
        write_tiles_by_row(cb_out_idx, out_gen, b * block_size, block_size, tile_bytes, block_size);
    }
}
