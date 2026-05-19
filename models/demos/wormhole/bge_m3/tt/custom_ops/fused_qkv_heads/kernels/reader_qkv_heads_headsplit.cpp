// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split reader for nlp_create_qkv_heads.
//
// Adapted from Qwen3-Embedding's reader_tm_tile_layout_nlp_create_qkv_heads_head_split.cpp
// (PR: ign/qwen3_0.6b_optimization).
//
// Difference vs Track A (batched): Track A processes one (batch, seq_tile) per
// work unit, reading 32 Q + 32 K + 32 V tiles from one S-tile-row. This kernel
// adds an inner head-group axis to the work split: each work unit processes
// just (heads_per_group) Q heads + (heads_per_group) K heads + same V heads
// for ONE (batch, seq_tile). For BGE-M3 (num_heads=16) with head_groups=16,
// each work unit reads only 2 Q + 2 K + 2 V tiles = 6 tiles, but there are
// 16 * 16 = 256 work units instead of 16 — covering the entire 110-core grid.
//
// Compile-time args:
//   0: q_heads_per_kv     (= num_q_heads / num_kv_heads; BGE: 1)
//   1: num_kv_heads       (BGE: 16)
//   2: head_dim_tiles     (= head_dim / TILE_W; BGE: 2)
//   3: in0_w_tiles        (= 3 * num_heads * head_dim_tiles; BGE: 96)
//   4: seq_tiles          (= seq_len / TILE_H; BGE: 16)
//   5: head_groups        (number of head groups per seq_tile; BGE: 16)
//   6: heads_per_group    (= num_kv_heads / head_groups; BGE: 1)
//   7+: TensorAccessorArgs for QKV-fused input tensor (in0_args)
//
// Runtime args:
//   0: in0_tensor_addr        (QKV-fused input base address)
//   1: num_work_units         (work units assigned to this core)
//   2: work_unit_start        (this core's first global work-unit index)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t q_heads_per_kv = get_compile_time_arg_val(0);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t head_groups = get_compile_time_arg_val(5);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(6);
    constexpr auto in0_args = TensorAccessorArgs<7>();

    constexpr uint32_t cb_id = 1;
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_id);

    constexpr uint32_t group_q_tiles = heads_per_group * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t group_kv_tiles = heads_per_group * head_dim_tiles;
    constexpr uint32_t q_tiles_total = num_kv_heads * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_dim_tiles;

    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t block = work_unit / head_groups;          // (batch, seq_tile) pair
        const uint32_t group = work_unit - block * head_groups;  // which head group
        const uint32_t s_tile = block % seq_tiles;
        const uint32_t batch = block / seq_tiles;
        const uint32_t block_base = batch * (seq_tiles * in0_w_tiles) + s_tile * in0_w_tiles;

        // ---- Q chunk ----
        // Q tile range inside the fused row: [group * group_q_tiles, group * group_q_tiles + group_q_tiles)
        const uint32_t q_offset_in_row = group * group_q_tiles;
        cb_reserve_back(cb_id, group_q_tiles);
        {
            uint32_t l1_write_addr = get_write_ptr(cb_id);
            const uint32_t q_base_tile = block_base + q_offset_in_row;
            for (uint32_t i = 0; i < group_q_tiles; ++i) {
                noc_async_read_tile(q_base_tile + i, s0, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, group_q_tiles);

        // ---- K chunk ----
        const uint32_t kv_offset_in_row = group * group_kv_tiles;
        cb_reserve_back(cb_id, group_kv_tiles);
        {
            uint32_t l1_write_addr = get_write_ptr(cb_id);
            const uint32_t k_base_tile = block_base + q_tiles_total + kv_offset_in_row;
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc_async_read_tile(k_base_tile + i, s0, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, group_kv_tiles);

        // ---- V chunk ----
        cb_reserve_back(cb_id, group_kv_tiles);
        {
            uint32_t l1_write_addr = get_write_ptr(cb_id);
            const uint32_t v_base_tile = block_base + q_tiles_total + kv_tiles_total + kv_offset_in_row;
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc_async_read_tile(v_base_tile + i, s0, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, group_kv_tiles);
    }
}
