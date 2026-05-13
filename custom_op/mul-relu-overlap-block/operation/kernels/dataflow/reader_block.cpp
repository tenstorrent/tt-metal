// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Block-based binary reader for mul-relu-overlap-block.
//
// Reads `num_tiles` tiles each from inputs A and B (DRAM-interleaved, same
// page count and dtype) and pushes them to circular buffers in fixed-size
// blocks of BS tiles. For the final (tail) block, only the real tiles are
// filled via NoC reads — the remaining slots in the BS-sized reservation
// are left uninitialized and pushed through unchanged. The compute kernel
// always sees full BS-tile blocks; the writer drops the unused tail slots,
// so the uninitialized data never reaches DRAM.
//
// Compile-time args:
//   [0] cb_a index
//   [1] cb_b index
//   [2] BS (block size, tiles)
//   [3..] chained TensorAccessorArgs(a, b)
//
// Runtime args:
//   [0] a_addr
//   [1] b_addr
//   [2] num_tiles
//   [3] start_tile_id

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a_idx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b_idx = get_compile_time_arg_val(1);
    constexpr uint32_t BS = get_compile_time_arg_val(2);

    constexpr auto a_args = TensorAccessorArgs<3, 0>();
    constexpr auto b_args =
        TensorAccessorArgs<a_args.next_compile_time_args_offset(), a_args.next_common_runtime_args_offset()>();

    const uint32_t tile_bytes_a = get_tile_size(cb_a_idx);
    const uint32_t tile_bytes_b = get_tile_size(cb_b_idx);

    const auto s_a = TensorAccessor(a_args, a_addr);
    const auto s_b = TensorAccessor(b_args, b_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_a(cb_a_idx);
    experimental::CircularBuffer cb_b(cb_b_idx);

    const uint32_t end_id = start_id + num_tiles;
    uint32_t tile_id = start_id;
    while (tile_id < end_id) {
        cb_a.reserve_back(BS);
        cb_b.reserve_back(BS);

        const uint32_t remaining = end_id - tile_id;
        const uint32_t tiles_this_block = remaining < BS ? remaining : BS;

        uint32_t off_a = 0;
        uint32_t off_b = 0;
        for (uint32_t i = 0; i < tiles_this_block; ++i) {
            noc.async_read(s_a, cb_a, tile_bytes_a, {.page_id = tile_id + i}, {.offset_bytes = off_a});
            noc.async_read(s_b, cb_b, tile_bytes_b, {.page_id = tile_id + i}, {.offset_bytes = off_b});
            off_a += tile_bytes_a;
            off_b += tile_bytes_b;
        }
        noc.async_read_barrier();

        cb_a.push_back(BS);
        cb_b.push_back(BS);

        tile_id += tiles_this_block;
    }
}
