// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split reader for nlp_concat_heads.
//
// Adapted from the Qwen3-Embedding head-split concat reader: each work unit
// processes a single (batch, seq_tile, head_group), reading
// `heads_per_group * head_dim_tiles` tiles. Combined with the head-split writer
// this expands the work-unit count from (batch * seq_tiles) to
// (batch * seq_tiles * head_groups), filling more of the compute grid.
//
// Compile-time args:
//   0: in0_h_tiles         (= seq_len / TILE_H, e.g. 16)
//   1: in0_w_tiles         (= head_dim / TILE_W, e.g. 2)
//   2: in0_c               (= num_heads, e.g. 16)
//   3: in0_HtWt            (= in0_h_tiles * in0_w_tiles, e.g. 32)
//   4: head_groups         (e.g. 16)
//   5: heads_per_group     (e.g. 1)
//   6+: TensorAccessorArgs for input [B, num_heads, S, head_dim]
//
// Runtime args:
//   0: in0_tensor_addr
//   1: num_work_units
//   2: work_unit_start

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t work_unit_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t in0_h_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in0_c = get_compile_time_arg_val(2);
    constexpr uint32_t in0_HtWt = get_compile_time_arg_val(3);
    constexpr uint32_t head_groups = get_compile_time_arg_val(4);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);
    constexpr auto in0_args = TensorAccessorArgs<6>();

    constexpr uint32_t cb_id = 0;
    constexpr uint32_t group_tiles = heads_per_group * in0_w_tiles;
    constexpr uint32_t in0_CHtWt = in0_c * in0_HtWt;

    const uint32_t tile_size_bytes = get_tile_size(cb_id);
    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);

    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t batch = block / in0_h_tiles;
        const uint32_t h_tile = block - batch * in0_h_tiles;
        const uint32_t head_start = group * heads_per_group;

        cb_reserve_back(cb_id, group_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id);

        // For each head in this group, read in0_w_tiles tiles at row h_tile.
        uint32_t row_base = batch * in0_CHtWt + head_start * in0_HtWt + h_tile * in0_w_tiles;
        for (uint32_t h = 0; h < heads_per_group; ++h) {
            for (uint32_t w_dim = 0; w_dim < in0_w_tiles; ++w_dim) {
                noc_async_read_tile(row_base + w_dim, s0, l1_write_addr);
                l1_write_addr += tile_size_bytes;
            }
            row_base += in0_HtWt;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, group_tiles);
    }
}
