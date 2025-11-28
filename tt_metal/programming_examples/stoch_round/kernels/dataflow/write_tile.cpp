// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_size_bytes);

    const uint32_t end_tile_id = start_tile_id + n_tiles;

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        noc_async_write_tile(i, dst, cb_out0_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
