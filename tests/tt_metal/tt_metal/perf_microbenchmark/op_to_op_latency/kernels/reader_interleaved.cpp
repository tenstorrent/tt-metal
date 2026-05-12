// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads `n_tiles` tiles from an interleaved DRAM buffer starting at
// `start_tile_id`, pushing them one-at-a-time into the input circular buffer
// (CB_in). One instance of this kernel runs on every Tensix core; each core
// gets its own [start_tile_id, start_tile_id + n_tiles) slice via runtime args.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t tile_size_bytes = get_tile_size(cb_in);
    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t end_tile_id = start_tile_id + n_tiles;
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        cb_reserve_back(cb_in, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_in);
        noc_async_read_tile(tile_id, src, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
