// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 Track A — batched writer for nlp_concat_heads (interleaved output).
//
// Drop-in replacement for
//   ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/
//       writer_unary_interleaved_start_id.cpp
// (which is what the stock concat program factory uses on the writer side).
//
// Difference vs stock unary writer: stock calls `noc.async_writes_flushed()`
// once per tile (lighter than a barrier but still serializes the per-tile
// dispatch). We collapse the loop into a single `noc_async_write_barrier()`
// at the end and pop the CB in one go. For a 32-tile-per-block workload this
// drops 32 per-tile flush points to one terminal barrier per block.
//
// Note: writer pages and reader pages are 1:1 — the reader scatters per-head
// reads into a single contiguous CB chunk, and the writer drains them into a
// concatenated output. So this is a straight tile copy with no remapping.
//
// Compile-time args:
//   0: block_tiles       (= num_heads * head_dim_tiles per block, e.g. 32)
//   1+: TensorAccessorArgs for the concatenated output tensor
//
// Runtime args:
//   0: out_tensor_addr
//   1: num_blocks
//   2: out_start_tile_id     (starting page id into the concat output)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ---- runtime args ----
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t out_tile_id = get_arg_val<uint32_t>(2);

    // ---- compile-time args ----
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id = 0;
    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    const auto s_out = TensorAccessor(out_args, out_tensor_addr);

    for (uint32_t block = 0; block < num_blocks; block++) {
        cb_wait_front(cb_id, block_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id);

        for (uint32_t i = 0; i < block_tiles; i++) {
            noc_async_write_tile(out_tile_id + i, s_out, l1_read_addr);
            l1_read_addr += tile_size_bytes;
        }
        out_tile_id += block_tiles;

        // ONE barrier per block instead of per-tile writes_flushed.
        noc_async_write_barrier();
        cb_pop_front(cb_id, block_tiles);
    }
}
