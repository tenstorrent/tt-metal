// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split writer for nlp_create_qkv_heads.
//
// Companion to reader_qkv_heads_headsplit.cpp. Writes (heads_per_group) Q heads
// + (heads_per_group) K heads + same V heads per work unit, with the destination
// tile-ID derived from (batch, seq_tile, group).
//
// Output layout for Q/K/V tensors is [B, num_heads, S, head_dim] in TILE_LAYOUT,
// so for tile-row `s_tile` of head `h` in batch `b`, the starting tile-id is:
//     batch_stride * b + HtWt * h + s_tile * head_dim_tiles
// where HtWt = seq_tiles * head_dim_tiles (per-head per-batch tile area).
//
// Compile-time args:
//   0: q_out_h_tiles        (= seq_tiles)
//   1: q_out_w_tiles        (= head_dim_tiles)
//   2: q_out_HtWt           (= seq_tiles * head_dim_tiles)
//   3: num_q_heads          (BGE: 16)
//   4: num_kv_heads         (BGE: 16)
//   5: q_heads_per_kv       (BGE: 1)
//   6: head_groups          (BGE: 16)
//   7: heads_per_group      (BGE: 1)
//   8: seq_tiles            (= seq_len / TILE_H; BGE: 16)
//   9+: TensorAccessorArgs for Q output
//   ...: TensorAccessorArgs for K output
//   ...: TensorAccessorArgs for V output
//
// Runtime args:
//   0: q_tensor_addr
//   1: k_tensor_addr
//   2: v_tensor_addr
//   3: num_work_units
//   4: work_unit_start

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_work_units = get_arg_val<uint32_t>(3);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t q_out_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t q_out_HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(3);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(5);
    constexpr uint32_t head_groups = get_compile_time_arg_val(6);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(7);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(8);
    constexpr auto q_args = TensorAccessorArgs<9>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id = 1;
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    // Device 2.0 data-movement API (see device_api_migration_guide.md).
    Noc noc;
    CircularBuffer cb(cb_id);

    constexpr uint32_t q_heads_per_group = heads_per_group * q_heads_per_kv;
    constexpr uint32_t group_q_tiles = q_heads_per_group * q_out_w_tiles;
    constexpr uint32_t group_kv_tiles = heads_per_group * q_out_w_tiles;
    constexpr uint32_t q_batch_stride = num_q_heads * q_out_HtWt;
    constexpr uint32_t kv_batch_stride = num_kv_heads * q_out_HtWt;

    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t s_tile = block % seq_tiles;
        const uint32_t batch = block / seq_tiles;

        // Starting heads for this work unit:
        const uint32_t q_head_start = group * q_heads_per_group;
        const uint32_t kv_head_start = group * heads_per_group;

        // ---- Q chunk ----
        cb.wait_front(group_q_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * q_batch_stride + q_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < q_heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb, sq, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(group_q_tiles);

        // ---- K chunk ----
        cb.wait_front(group_kv_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * kv_batch_stride + kv_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb, sk, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(group_kv_tiles);

        // ---- V chunk ----
        cb.wait_front(group_kv_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * kv_batch_stride + kv_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb, sv, tile_size_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += tile_size_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb.pop_front(group_kv_tiles);
    }
}
