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
        get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    const uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

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
    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // query width in tiles
    constexpr uint32_t kWt = get_compile_time_arg_val(1);              // key/value width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // heads per group

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_output_args = TensorAccessorArgs<5>();
    constexpr auto attn_output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto query_args = TensorAccessorArgs<attn_output_args.next_compile_time_args_offset()>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1U;
    constexpr uint32_t num_of_interm_tiles = 2U;

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);

    // Create TensorAccessor generators
    const auto grad_output_address_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto attn_output_address_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
    const auto query_address_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_address_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_address_generator = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);
    const auto intermediates_address_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);

    generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce (auto-detects data type)

    const uint32_t num_of_groups = q_heads / heads_per_group;

    // process rows of K and V assigned to this core
    // stream rows from Q, dO, O(for all heads associated with this group of K and V)
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t global_row_idx = start_row + i;
        const uint32_t kv_start_idx = global_row_idx * kWt;

        read_tiles_by_row(cb_key, key_address_generator, kv_start_idx, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, kv_start_idx, kWt, tile_bytes, kWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;  // which group of K and V we are processing
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);  // which batch we are processing

        // the index of the first head in Q associated with this group of K and V
        const uint32_t first_q_head_idx = group_idx * heads_per_group;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;

        // Mask is (1, 1, S, S) - same mask for all batches/heads, indexed by sequence position only
        // For KV kernel, we read column (global_row_idx % Ht) from each row h of the mask
        const uint32_t mask_offset = (global_row_idx % Ht);

        // add change here: multiply by num_of_interm_tiles because we need to read 2 tiles per head row
        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        // TODO: add calculation for dO, O indexes because in forward pass they are stored with shape (B, 1, S,
        // qNH*qEmbd)
        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
            for (uint32_t h = 0; h < Ht; ++h) {
                const uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_tiles_by_row(cb_query, query_address_generator, q_start_idx, qWt, tile_bytes, qWt);

                // read one tile of attn_mask for current row of K and V
                // row of K define the column in (QK^T) matrix, so it define the column of attn_mask
                read_one_tile(cb_attn_mask, mask_address_generator, mask_offset + h * Ht);

                // Read intermediates - one tile per row (contains 1/sum_exp values from forward pass)
                // TODO[improve](vmelnykov): Now we share two intermediates values per head row: row-wise max value and
                // 1/sum_exp In future we can think about optimizing this by sharing logsumexp only
                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_idx,
                    num_of_interm_tiles,
                    tile_bytes,
                    num_of_interm_tiles);

                read_tiles_by_row(cb_grad_output, grad_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);
                read_tiles_by_row(cb_attn_output, attn_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);
            }
            // update offset for intermediates only (mask is (1,1,S,S) - same for all heads)
            intermediates_offset += Ht * num_of_interm_tiles;  // jump to the head in intermediates
        }
    }
}
