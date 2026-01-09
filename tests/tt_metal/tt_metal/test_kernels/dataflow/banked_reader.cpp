// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    std::uint32_t src_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<uint32_t>(1);

    experimental::CircularBuffer cb(cb_id);
    experimental::Noc noc(noc_index);

    const uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = cb.get_tile_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base, page_size);

    // read tiles from src to CB
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        cb.reserve_back(ublock_size_tiles);
        noc.async_read(tensor_accessor, cb, tile_bytes, {.page_id = i}, {});
        noc.async_read_barrier();
        cb.push_back(ublock_size_tiles);
    }
}
