// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in_accessor = TensorAccessor(in_args, in_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t cb_addr = get_write_ptr(cb_in);
        noc_async_read_tile(start_tile_id + i, in_accessor, cb_addr);  // Offset by start_tile_id
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
