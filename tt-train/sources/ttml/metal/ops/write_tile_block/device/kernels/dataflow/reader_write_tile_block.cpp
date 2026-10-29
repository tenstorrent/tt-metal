// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Reads this core's tile range [start_tile, start_tile + num_tiles) of the source into the CB, block_size tiles at a
// time (a short last block still pushes block_size slots; the writer pops the same).
void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t source_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto source_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_idx);
    const auto source = TensorAccessor(source_args, source_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t += block_size) {
        const uint32_t current = (t + block_size <= num_tiles) ? block_size : (num_tiles - t);
        read_tiles_by_row(cb_idx, source, start_tile + t, current, tile_bytes, block_size);
    }
}
