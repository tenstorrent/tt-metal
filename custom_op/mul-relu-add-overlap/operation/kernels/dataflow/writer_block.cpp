// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Block-based writer for mul-relu-add-overlap. Pops BS tiles per CB
// transaction (matching the block size used by the compute kernel) but only
// issues NoC writes for the `tiles_this_block <= BS` real output tiles. On
// the tail block the unused slots are dropped without being written to DRAM.
//
// Compile-time args:
//   [0] cb_out index
//   [1] BS (block size, tiles)
//   [2..] TensorAccessorArgs(out)
//
// Runtime args:
//   [0] out_addr
//   [1] num_tiles
//   [2] start_tile_id

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out_idx = get_compile_time_arg_val(0);
    constexpr uint32_t BS = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2, 0>();

    const uint32_t tile_bytes = get_tile_size(cb_out_idx);
    const auto s = TensorAccessor(out_args, out_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_out_idx);

    const uint32_t end_id = start_id + num_tiles;
    uint32_t tile_id = start_id;
    while (tile_id < end_id) {
        cb_out.wait_front(BS);

        const uint32_t remaining = end_id - tile_id;
        const uint32_t tiles_this_block = remaining < BS ? remaining : BS;

        uint32_t off = 0;
        for (uint32_t i = 0; i < tiles_this_block; ++i) {
            noc.async_write(cb_out, s, tile_bytes, {.offset_bytes = off}, {.page_id = tile_id + i});
            off += tile_bytes;
        }
        noc.async_write_barrier();

        cb_out.pop_front(BS);
        tile_id += tiles_this_block;
    }
}
