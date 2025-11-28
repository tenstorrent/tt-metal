// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, tile_size_bytes);

    const uint32_t end_tile_id = start_tile_id + n_tiles;

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(i, src, cb_in0_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);
    }
}
