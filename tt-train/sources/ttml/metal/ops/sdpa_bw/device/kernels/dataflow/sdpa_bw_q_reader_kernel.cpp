// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);    // grad_output buffer address
    const uint32_t attn_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);    // attn_output buffer address
    const uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);          // query buffer address
    const uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);            // key buffer address
    const uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);          // value buffer address
    const uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // mask buffer address
    const uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);  // intermediates buffer address
    const uint32_t num_rows_to_process =
        get_arg_val<uint32_t>(runtime_args_counter++);                         // rows to process in this kernel
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);  // starting row for this core

    // Circular buffer indices
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
    constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_7;

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // query width in tiles (also kWt, vWt)
    constexpr uint32_t Ht = get_compile_time_arg_val(1);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(2);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(3);  // heads per group

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_output_args = TensorAccessorArgs<4>();
    constexpr auto attn_output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto query_args = TensorAccessorArgs<attn_output_args.next_compile_time_args_offset()>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1U;
    generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce (auto-detects data type)

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);

    // Create TensorAccessor generators for inputs
    const auto grad_output_addr_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto attn_output_addr_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
    const auto query_addr_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_addr_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_addr_generator = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto mask_addr_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);

    const uint32_t num_of_groups = q_heads / heads_per_group;
    const uint32_t num_of_interm_tiles = 2U;

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t global_row_idx = start_row + i;
        const uint32_t q_start_idx = global_row_idx * qWt;
        // Read attn_output row
        read_tiles_by_row(cb_attn_output, attn_output_addr_generator, q_start_idx, qWt, tile_bytes, qWt);
        // Read grad_output row
        read_tiles_by_row(cb_grad_output, grad_output_addr_generator, q_start_idx, qWt, tile_bytes, qWt);
        // Read query row
        read_tiles_by_row(cb_query, query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);

        const uint32_t q_head_idx = (global_row_idx / Ht) % q_heads;  // which head of Q we are processing right now

        // which batch we are processing right now
        const uint32_t batch_idx = global_row_idx / (Ht * q_heads);
        // calculate which group of K and V we need to read for this head of Q
        const uint32_t kv_group_idx = q_head_idx / heads_per_group;

        // calculate the starting index of K and V to read
        const uint32_t kv_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * qWt;

        // Mask is (1, 1, S, S) - same mask for all batches/heads, indexed by sequence position only
        const uint32_t mask_offset = (global_row_idx % Ht) * Ht;

        // read intermediates for current row of Q
        // intermediates shape: (B, qNH, S, 64) -> (batch, heads, seq_len, 2 tiles)
        // TODO[improve](vmelnykov): Now we share two intermediates values per head row: row-wise max value and
        // 1/sum_exp In future we can think about optimizing this by sharing logsumexp only
        const uint32_t intermediates_idx = global_row_idx * num_of_interm_tiles;
        read_tiles_by_row(
            cb_intermediates,
            intermediates_addr_generator,
            intermediates_idx,
            num_of_interm_tiles,
            tile_bytes,
            num_of_interm_tiles);

        for (uint32_t h = 0; h < Ht; ++h) {
            const uint32_t kv_start_idx =
                kv_offset + h * qWt;  // jump to the next row of K and V, qWt == kWt == vWt(same embedding size)

            // Read one row of K and V
            read_tiles_by_row(cb_key, key_addr_generator, kv_start_idx, qWt, tile_bytes, qWt);

            // read one tile of attn_mask for current row of K and V
            // row of K define the column in (QK^T) matrix, so it define the column of attn_mask to read
            read_one_tile(cb_attn_mask, mask_addr_generator, mask_offset + h);

            read_tiles_by_row(cb_value, value_addr_generator, kv_start_idx, qWt, tile_bytes, qWt);
        }
    }
}
