// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split reader, K-BF4 variant.
//
// Identical to reader_qkv_heads_headsplit.cpp except K tiles are pushed to a
// SEPARATE input CB (cb_k_in) so a compute kernel can typecast them BF8->BF4
// before the writer stores them. Q and V still use the shared CB (cb_qv).
//
// Compile-time args:
//   0: q_heads_per_kv     (BGE: 1)
//   1: num_kv_heads       (BGE: 16)
//   2: head_dim_tiles     (BGE: 2)
//   3: in0_w_tiles        (= 3 * num_heads * head_dim_tiles; BGE: 96)
//   4: seq_tiles          (BGE: 16)
//   5: head_groups        (BGE: 16)
//   6: heads_per_group    (BGE: 1)
//   7: cb_qv              (shared Q/V CB index)
//   8: cb_k_in            (BF8 K input CB index)
//   9+: TensorAccessorArgs for QKV-fused input tensor
//
// Runtime args:
//   0: in0_tensor_addr
//   1: num_work_units
//   2: work_unit_start

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

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
    constexpr uint32_t cb_qv = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(8);
    constexpr auto in0_args = TensorAccessorArgs<9>();

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_qv);

    Noc noc;
    CircularBuffer cb_qv_h(cb_qv);
    CircularBuffer cb_k_h(cb_k_in);

    constexpr uint32_t group_q_tiles = heads_per_group * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t group_kv_tiles = heads_per_group * head_dim_tiles;
    constexpr uint32_t q_tiles_total = num_kv_heads * q_heads_per_kv * head_dim_tiles;
    constexpr uint32_t kv_tiles_total = num_kv_heads * head_dim_tiles;

    for (uint32_t w = 0; w < num_work_units; ++w) {
        const uint32_t work_unit = work_unit_start + w;
        const uint32_t block = work_unit / head_groups;
        const uint32_t group = work_unit - block * head_groups;
        const uint32_t s_tile = block % seq_tiles;
        const uint32_t batch = block / seq_tiles;
        const uint32_t block_base = batch * (seq_tiles * in0_w_tiles) + s_tile * in0_w_tiles;

        // ---- Q chunk -> shared CB ----
        const uint32_t q_offset_in_row = group * group_q_tiles;
        cb_qv_h.reserve_back(group_q_tiles);
        {
            uint32_t l1_write_offset = 0;
            const uint32_t q_base_tile = block_base + q_offset_in_row;
            for (uint32_t i = 0; i < group_q_tiles; ++i) {
                noc.async_read(
                    s0, cb_qv_h, tile_size_bytes, {.page_id = q_base_tile + i}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
            }
        }
        noc.async_read_barrier();
        cb_qv_h.push_back(group_q_tiles);

        // ---- K chunk -> separate BF8 K-input CB (typecast happens in compute) ----
        const uint32_t kv_offset_in_row = group * group_kv_tiles;
        cb_k_h.reserve_back(group_kv_tiles);
        {
            uint32_t l1_write_offset = 0;
            const uint32_t k_base_tile = block_base + q_tiles_total + kv_offset_in_row;
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc.async_read(
                    s0, cb_k_h, tile_size_bytes, {.page_id = k_base_tile + i}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
            }
        }
        noc.async_read_barrier();
        cb_k_h.push_back(group_kv_tiles);

        // ---- V chunk -> shared CB ----
        cb_qv_h.reserve_back(group_kv_tiles);
        {
            uint32_t l1_write_offset = 0;
            const uint32_t v_base_tile = block_base + q_tiles_total + kv_tiles_total + kv_offset_in_row;
            for (uint32_t i = 0; i < group_kv_tiles; ++i) {
                noc.async_read(
                    s0, cb_qv_h, tile_size_bytes, {.page_id = v_base_tile + i}, {.offset_bytes = l1_write_offset});
                l1_write_offset += tile_size_bytes;
            }
        }
        noc.async_read_barrier();
        cb_qv_h.push_back(group_kv_tiles);
    }
}
