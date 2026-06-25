// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // Tile offset for multi-core

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_size_bytes);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t cb_addr = get_read_ptr(cb_out);
        noc_async_write_tile(start_tile_id + i, out_accessor, cb_addr);  // Offset by start_tile_id
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
