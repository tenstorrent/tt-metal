// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Writes this core's source tiles [start_tile, start_tile + num_tiles) to their destination tiles
// (tile_row_offset + i / Wt_src) * Wt_dst + tile_col_offset + i % Wt_src, mirroring the reader's blocks; a
// full-width source is one contiguous range, a narrower one a strided column block.
void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t destination_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_tile = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t tile_row_offset = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t tile_col_offset = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t Wt_src = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t Wt_dst = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto destination_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_idx);
    const auto destination = TensorAccessor(destination_args, destination_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t += block_size) {
        const uint32_t current = (t + block_size <= num_tiles) ? block_size : (num_tiles - t);
        cb_wait_front(cb_idx, block_size);
        uint32_t l1_read_addr = get_read_ptr(cb_idx);
        for (uint32_t i = 0; i < current; ++i) {
            const uint32_t src_tile = start_tile + t + i;
            const uint32_t dst_tile =
                (tile_row_offset + src_tile / Wt_src) * Wt_dst + tile_col_offset + src_tile % Wt_src;
            noc_async_write_page(dst_tile, destination, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_idx, block_size);
    }
}
