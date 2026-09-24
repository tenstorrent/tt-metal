// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 Track A — batched-barrier writer for nlp_create_qkv_heads
// (no TRANSPOSE_K_HEADS, no input_tensor_kv variant).
//
// Replaces ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/
// device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp.
//
// Difference vs stock: stock performs a one-tile CB wait/write/barrier/pop
// sequence for each tile (96 barriers per block-of-96-tiles). This kernel waits
// on a whole Q/K/V chunk at once, fires all the async writes, then issues a
// single barrier per chunk (3 barriers per block).
//
// Compile-time args:
//   0: q_out_h_tiles       (= S / TILE_H, e.g. 16)
//   1: q_out_w_tiles       (= head_dim / TILE_W, e.g. 2)
//   2: q_out_HtWt          (= q_out_h_tiles * q_out_w_tiles, e.g. 32)
//   3: num_q_heads         (e.g. 16)
//   4: num_kv_heads        (e.g. 16)
//   5+: TensorAccessorArgs for Q output
//   5+N: TensorAccessorArgs for K output
//   5+2N: TensorAccessorArgs for V output
//
// Runtime args (matching stock layout):
//   0: q_tensor_addr
//   1: k_tensor_addr
//   2: v_tensor_addr
//   3: num_blocks
//   4: q_out_h_dim          (the starting h-tile index in the output's S dim)
//   5: q_out_tensor_tile_id
//   6: k_out_tensor_tile_id
//   7: v_out_tensor_tile_id

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // ---- runtime args ----
    uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t q_out_h_dim = get_arg_val<uint32_t>(4);
    uint32_t q_out_tile_id = get_arg_val<uint32_t>(5);
    uint32_t k_out_tile_id = get_arg_val<uint32_t>(6);
    uint32_t v_out_tile_id = get_arg_val<uint32_t>(7);

    // ---- compile-time args ----
    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(3);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    // Derived tile counts per block (one S tile-row).
    constexpr uint32_t q_chunk_tiles = num_q_heads * q_out_w_tiles;
    constexpr uint32_t kv_chunk_tiles = num_kv_heads * q_out_w_tiles;

    // CB shared with reader.
    constexpr uint32_t cb_id = 1;
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    // Device 2.0 data-movement API (see device_api_migration_guide.md).
    Noc noc;
    CircularBuffer cb(cb_id);

    // ---- main loop ----
    for (uint32_t block = 0; block < num_blocks; block++) {
        // ---- Q chunk ----
        cb.wait_front(q_chunk_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = q_out_tile_id;  // starts at this h_dim's row in head 0
            for (uint32_t c_dim = 0; c_dim < num_q_heads; c_dim++) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    noc.async_write(cb, sq, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(q_chunk_tiles);

        // ---- K chunk (same layout, kv_num_heads × w_tiles) ----
        cb.wait_front(kv_chunk_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = k_out_tile_id;
            for (uint32_t c_dim = 0; c_dim < num_kv_heads; c_dim++) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    noc.async_write(cb, sk, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(kv_chunk_tiles);

        // ---- V chunk ----
        cb.wait_front(kv_chunk_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = v_out_tile_id;
            for (uint32_t c_dim = 0; c_dim < num_kv_heads; c_dim++) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                    noc.async_write(cb, sv, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(kv_chunk_tiles);

        // ---- Advance to the next h_dim within this batch (mirror stock semantics) ----
        q_out_h_dim++;
        if (q_out_h_dim < q_out_h_tiles) {
            q_out_tile_id += q_out_w_tiles;
            k_out_tile_id += q_out_w_tiles;
            v_out_tile_id += q_out_w_tiles;
        } else {
            // Roll over to next batch: jump to start of next batch's first head row.
            // For Q: q_out_tile_id should advance by (num_q_heads-1)*q_out_HtWt + q_out_w_tiles
            //        from the LAST head row to land at the next batch's head 0 row 0.
            // We replicate the stock writer's tile-id update for full correctness.
            uint32_t q_last_head_row = q_out_tile_id + (num_q_heads - 1) * q_out_HtWt;
            uint32_t k_last_head_row = k_out_tile_id + (num_kv_heads - 1) * q_out_HtWt;
            uint32_t v_last_head_row = v_out_tile_id + (num_kv_heads - 1) * q_out_HtWt;
            q_out_tile_id = q_last_head_row + q_out_w_tiles;
            k_out_tile_id = k_last_head_row + q_out_w_tiles;
            v_out_tile_id = v_last_head_row + q_out_w_tiles;
            q_out_h_dim = 0;
        }
    }
}
