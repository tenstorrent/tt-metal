// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t query_address = get_arg_val<uint32_t>(runtime_args_counter++);  // query buffer address
    uint32_t key_address = get_arg_val<uint32_t>(runtime_args_counter++);    // key buffer address
    uint32_t value_address = get_arg_val<uint32_t>(runtime_args_counter++);  // value buffer address
    uint32_t mask_address = get_arg_val<uint32_t>(runtime_args_counter++);   // mask buffer address
    uint32_t gate_address = get_arg_val<uint32_t>(runtime_args_counter++);   // gate buffer address
    // Then the core's work items (Q chunks): num_runs, (first_item, num_items) per run.
    const uint32_t num_runs = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
#endif
#ifdef HAS_GATE
    constexpr uint32_t cb_gate = tt::CBIndex::c_16;
#endif

    constexpr uint32_t qWt = get_compile_time_arg_val(0);      // num tile in inner dim in query/key (d_qk/TILE_W)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);      // num tile in inner dim in value (d_v/TILE_W)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);       // (S / TILE_H)
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);  // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // num of heads per group
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);       // K/V tiles per inner-loop step
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);       // query tile-rows per Q chunk
    constexpr uint32_t q_num_chunks = Ht / Sq_chunk_t;
    constexpr auto query_args = TensorAccessorArgs<7>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
#ifdef HAS_GATE
    constexpr auto gate_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
#endif

    const uint32_t tile_bytes = get_tile_size(cb_query);

    const auto query_address_generator = TensorAccessor(query_args, query_address);
    const auto key_address_generator = TensorAccessor(key_args, key_address);
    const auto value_address_generator = TensorAccessor(value_args, value_address);

#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address);
#endif
#ifdef HAS_GATE
    const auto gate_address_generator = TensorAccessor(gate_args, gate_address);
#endif

    // Note: Tile generation (reduction_scaler, matmul_reduce, causal_mask) moved to writer kernel
    // This allows reader to start DRAM reads immediately

    const uint32_t num_of_groups = q_heads / heads_per_group;

    // Reads one Q chunk (Sq_chunk_t query tile-rows) and then the K/V tiles it attends to, in steps of
    // Sk_chunk_t tiles. The push order must match the cb_wait_front order in the compute kernel.
    auto read_q_chunk = [&](const uint32_t global_q_chunk_idx) {
        const uint32_t seq_idx = global_q_chunk_idx / q_num_chunks;  // batch * q_heads + head
        const uint32_t first_row_tile = (global_q_chunk_idx % q_num_chunks) * Sq_chunk_t;
        const uint32_t global_first_row = seq_idx * Ht + first_row_tile;

        read_tiles_by_row(
            cb_query, query_address_generator, global_first_row * qWt, Sq_chunk_t * qWt, tile_bytes, Sq_chunk_t * qWt);

        const uint32_t q_head_idx = seq_idx % q_heads;
        const uint32_t batch_idx = seq_idx / q_heads;
        const uint32_t kv_group_idx = q_head_idx / heads_per_group;
        const uint32_t key_offset = (batch_idx * num_of_groups + kv_group_idx) * qWt * Ht;
        const uint32_t value_offset = (batch_idx * num_of_groups + kv_group_idx) * vWt * Ht;

#ifdef CAUSAL_MASK
        const uint32_t num_kv_tiles_to_read = round_up(first_row_tile + Sq_chunk_t, Sk_chunk_t);
#else
        const uint32_t num_kv_tiles_to_read = Ht;
#endif
        const uint32_t num_kv_steps = num_kv_tiles_to_read / Sk_chunk_t;

        for (uint32_t k_step = 0; k_step < num_kv_steps; ++k_step) {
            const uint32_t h_start = k_step * Sk_chunk_t;
            const uint32_t key_start_idx = key_offset + h_start * qWt;
            // K lands feature-major in cb_key (all Sk_chunk_t tiles of feature 0, then feature 1, ...),
            // the order matmul_block walks when it multiplies by K^T.
            read_tile_block_transposed(cb_key, key_address_generator, key_start_idx, Sk_chunk_t, qWt, tile_bytes);

#ifdef USE_ATTN_MASK
            // Arbitrary mask: Sq_chunk_t x Sk_chunk_t tiles of this step, row-major.
            {
                constexpr uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
                cb_reserve_back(cb_attn_mask, mask_tiles);
                uint32_t l1_addr = get_write_ptr(cb_attn_mask);
                for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                    uint32_t src_tile_id = (first_row_tile + r) * Ht + h_start;
                    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
                        noc_async_read_page(src_tile_id, mask_address_generator, l1_addr);
                        src_tile_id += 1U;
                        l1_addr += tile_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_attn_mask, mask_tiles);
            }
#endif
            // Note: For CAUSAL_MASK, the two reusable mask tiles are generated once by the writer.
            const uint32_t value_start_idx = value_offset + h_start * vWt;
            read_tiles_by_row(
                cb_value, value_address_generator, value_start_idx, Sk_chunk_t * vWt, tile_bytes, Sk_chunk_t * vWt);
        }
#ifdef HAS_GATE
        read_tiles_by_row(
            cb_gate, gate_address_generator, global_first_row * vWt, Sq_chunk_t * vWt, tile_bytes, Sq_chunk_t * vWt);
#endif
    };

    for (uint32_t run = 0; run < num_runs; ++run) {
        const uint32_t first_item = get_arg_val<uint32_t>(runtime_args_counter++);
        const uint32_t num_items = get_arg_val<uint32_t>(runtime_args_counter++);
        for (uint32_t item = first_item; item < first_item + num_items; ++item) {
            read_q_chunk(item);
        }
    }
}
