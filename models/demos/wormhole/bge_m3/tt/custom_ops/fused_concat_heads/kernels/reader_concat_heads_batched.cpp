// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 Track A — batched-barrier reader for nlp_concat_heads
// (interleaved input, single output tensor, no transpose).
//
// Replaces ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads/
// device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp.
//
// Difference vs stock: stock performs a one-tile CB reserve/read/barrier/push
// sequence for each tile (num_heads * head_dim_tiles barriers per block). This
// kernel issues a single reserve/barrier/push per Q,K,V-style block.
//
// For BGE-M3 B1/S512 each block is 16 heads * 2 head_dim_tiles = 32 tiles,
// dropping the reader's per-block NoC sync count from 32 to 1.
//
// Compile-time args (match stock order so the kernel binary's accessor
// indexes align with what the Python wrapper passes):
//   0: in0_h_tiles         (= seq_len / TILE_H, e.g. 16)
//   1: in0_w_tiles         (= head_dim / TILE_W, e.g. 2)
//   2: in0_c               (= num_heads, e.g. 16)
//   3: in0_HtWt            (= in0_h_tiles * in0_w_tiles, e.g. 32)
//   4+: TensorAccessorArgs for the head-laid-out input tensor
//
// Runtime args (match stock layout):
//   0: in0_tensor_addr
//   1: num_blocks
//   2: in0_h_dim                (S-tile-row index within this batch)
//   3: in0_tensor_tile_id       (starting page id, head 0 row in0_h_dim col 0)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // ---- runtime args ----
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t in0_h_dim = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(3);

    // ---- compile-time args ----
    constexpr uint32_t in0_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in0_c = get_compile_time_arg_val(2);
    constexpr uint32_t in0_HtWt = get_compile_time_arg_val(3);
    constexpr auto in0_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t block_tiles = in0_c * in0_w_tiles;

    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    // Device 2.0 data-movement API (see device_api_migration_guide.md).
    Noc noc;
    CircularBuffer cb(cb_id);

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Read the whole concat block (num_heads * head_dim_tiles) in one go.
        cb.reserve_back(block_tiles);
        uint32_t l1_write_offset = 0;

        uint32_t tile_id_along_c = in0_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
            uint32_t tid = tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
                noc.async_read(s0, cb, tile_size_bytes, {.page_id = tid}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
                tid++;
            }
            tile_id_along_c += in0_HtWt;
        }
        // After the last head's last tile, `tile_id_along_c` points at
        // (in0_tensor_tile_id + in0_c * in0_HtWt), which is the same value
        // the stock reader uses to roll over to the next batch.
        uint32_t in0_tensor_current_tile_id_after = tile_id_along_c;

        // ONE barrier for the whole block instead of one-per-tile.
        noc.async_read_barrier();
        cb.push_back(block_tiles);

        // Mirror stock tile-id advance.
        in0_h_dim++;
        if (in0_h_dim < in0_h_tiles) {
            in0_tensor_tile_id += in0_w_tiles;
        } else {
            in0_tensor_tile_id = in0_tensor_current_tile_id_after;
            in0_h_dim = 0;
        }
    }
}
