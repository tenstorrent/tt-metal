// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split writer, K-BF4 variant.
//
// Identical to writer_qkv_heads_headsplit.cpp except K tiles are read from the
// BF4 output CB (cb_k_out, produced by compute_k_typecast_bf4.cpp) and written
// to the BF4 K output tensor. Q and V come from the shared CB (cb_qv, BF8).
//
// The shared CB carries Q then V per work unit (reader push order), so this
// writer pops Q from cb_qv, then K from cb_k_out, then V from cb_qv.
//
// Compile-time args:
//   0: q_out_h_tiles
//   1: q_out_w_tiles
//   2: q_out_HtWt
//   3: num_q_heads
//   4: num_kv_heads
//   5: q_heads_per_kv
//   6: head_groups
//   7: heads_per_group
//   8: seq_tiles
//   9: cb_qv           (shared Q/V CB index, BF8)
//   10: cb_k_out       (BF4 K CB index)
//   11+: TensorAccessorArgs for Q output
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
    constexpr uint32_t cb_qv = get_compile_time_arg_val(9);
    constexpr uint32_t cb_k_out = get_compile_time_arg_val(10);
    constexpr auto q_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const uint32_t qv_tile_bytes = get_tile_size(cb_qv);
    const uint32_t k_tile_bytes = get_tile_size(cb_k_out);

    const auto sq = TensorAccessor(q_args, q_tensor_addr);
    const auto sk = TensorAccessor(k_args, k_tensor_addr);
    const auto sv = TensorAccessor(v_args, v_tensor_addr);

    Noc noc;
    CircularBuffer cb_qv_h(cb_qv);
    CircularBuffer cb_k_h(cb_k_out);

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

        const uint32_t q_head_start = group * q_heads_per_group;
        const uint32_t kv_head_start = group * heads_per_group;

        // ---- Q chunk (shared CB, BF8) ----
        cb_qv_h.wait_front(group_q_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * q_batch_stride + q_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < q_heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb_qv_h, sq, qv_tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += qv_tile_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb_qv_h.pop_front(group_q_tiles);

        // ---- K chunk (BF4 CB from compute) ----
        cb_k_h.wait_front(group_kv_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * kv_batch_stride + kv_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb_k_h, sk, k_tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += k_tile_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb_k_h.pop_front(group_kv_tiles);

        // ---- V chunk (shared CB, BF8) ----
        cb_qv_h.wait_front(group_kv_tiles);
        {
            uint32_t l1_read_offset = 0;
            uint32_t row_base = batch * kv_batch_stride + kv_head_start * q_out_HtWt + s_tile * q_out_w_tiles;
            for (uint32_t h = 0; h < heads_per_group; ++h) {
                uint32_t dst = row_base;
                for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; ++w_dim) {
                    noc.async_write(cb_qv_h, sv, qv_tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = dst});
                    l1_read_offset += qv_tile_bytes;
                    dst++;
                }
                row_base += q_out_HtWt;
            }
        }
        noc.async_write_barrier();
        cb_qv_h.pop_front(group_kv_tiles);
    }
}
