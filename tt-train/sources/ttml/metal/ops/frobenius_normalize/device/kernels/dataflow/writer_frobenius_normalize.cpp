// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_output = tt::CBIndex::c_5;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr auto output_args = TensorAccessorArgs<1>();

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto output_addr_gen = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx += block_size) {
        const uint32_t current = std::min(block_size, num_tiles - tile_idx);
        write_tiles_by_row(cb_output, output_addr_gen, start_tile_id + tile_idx, current, tile_bytes, current);
    }
}
