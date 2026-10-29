// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Writes this core's tiles to destination tiles [tile_offset + start_tile, ...), mirroring the reader's blocks.
void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t destination_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto destination_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_idx);
    const auto destination = TensorAccessor(destination_args, destination_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t += block_size) {
        const uint32_t current = (t + block_size <= num_tiles) ? block_size : (num_tiles - t);
        write_tiles_by_row(cb_idx, destination, tile_offset + start_tile + t, current, tile_bytes, block_size);
    }
}
