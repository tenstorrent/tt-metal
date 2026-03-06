// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
#ifndef USE_PRECOMPUTED_U_SCALER
    const uint32_t attn_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
#endif
    const uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
#ifdef USE_PRECOMPUTED_U_SCALER
    const uint32_t u_scaler_addr = get_arg_val<uint32_t>(runtime_args_counter++);
#endif
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
#ifndef USE_PRECOMPUTED_U_SCALER
    constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;
#endif
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
#endif
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_7;
#ifdef USE_PRECOMPUTED_U_SCALER
    constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_14;
#endif

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);
    constexpr uint32_t kWt = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_output_args = TensorAccessorArgs<5>();
#ifdef USE_PRECOMPUTED_U_SCALER
    constexpr auto query_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
#else
    constexpr auto attn_output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto query_args = TensorAccessorArgs<attn_output_args.next_compile_time_args_offset()>();
#endif
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
#ifdef USE_PRECOMPUTED_U_SCALER
    constexpr auto u_scaler_args = TensorAccessorArgs<intermediates_args.next_compile_time_args_offset()>();
#endif

    constexpr uint32_t onetile = 1U;
    constexpr uint32_t num_of_interm_tiles = 2U;

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);

    // Create TensorAccessor generators
    const auto grad_output_address_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
#ifndef USE_PRECOMPUTED_U_SCALER
    const auto attn_output_address_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
#endif
    const auto query_address_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_address_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_address_generator = TensorAccessor(value_args, value_addr, tile_bytes);
#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);
#endif
    const auto intermediates_address_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);
#ifdef USE_PRECOMPUTED_U_SCALER
    const uint32_t u_scaler_tile_bytes = get_tile_size(cb_u_scalar_row);
    const auto u_scaler_address_generator = TensorAccessor(u_scaler_args, u_scaler_addr, u_scaler_tile_bytes);
#endif

    generate_matmul_row_reduce_tile(cb_matmul_reduce);

    const uint32_t num_of_groups = q_heads / heads_per_group;

#ifdef BALANCED_PARALLELISM
    constexpr uint32_t pairs_per_seq = Ht / 2;

    auto read_row = [&](const uint32_t global_row_idx) {
        const uint32_t kv_start_idx = global_row_idx * kWt;

        read_tiles_by_row(cb_key, key_address_generator, kv_start_idx, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, kv_start_idx, kWt, tile_bytes, kWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);

        const uint32_t first_q_head_idx = group_idx * heads_per_group;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;

        const uint32_t k_row_tile = global_row_idx % Ht;

        const uint32_t q_start_tile = k_row_tile;
        const uint32_t num_q_tiles_to_read = Ht - k_row_tile;

        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
#ifdef USE_PRECOMPUTED_U_SCALER
            const uint32_t u_scaler_head_offset = (batch_idx * q_heads + first_q_head_idx + q_head_idx) * Ht;
#endif

            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_read; ++q_idx) {
                const uint32_t h = q_start_tile + q_idx;

                const uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_tiles_by_row(cb_query, query_address_generator, q_start_idx, qWt, tile_bytes, qWt);

                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_idx,
                    num_of_interm_tiles,
                    tile_bytes,
                    num_of_interm_tiles);

                read_tiles_by_row(cb_grad_output, grad_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef USE_PRECOMPUTED_U_SCALER
                read_one_tile(cb_u_scalar_row, u_scaler_address_generator, u_scaler_head_offset + h);
#else
                read_tiles_by_row(cb_attn_output, attn_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);
#endif
            }
            intermediates_offset += Ht * num_of_interm_tiles;
        }
    };

    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        read_row(light_global_row);
        read_row(heavy_global_row);
    }
#else
    // process rows of K and V assigned to this core
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t global_row_idx = start_row + i;
        const uint32_t kv_start_idx = global_row_idx * kWt;

        read_tiles_by_row(cb_key, key_address_generator, kv_start_idx, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, kv_start_idx, kWt, tile_bytes, kWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);

        const uint32_t first_q_head_idx = group_idx * heads_per_group;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;

        const uint32_t k_row_tile = global_row_idx % Ht;

#ifdef CAUSAL_MASK
        const uint32_t q_start_tile = k_row_tile;
        const uint32_t num_q_tiles_to_read = Ht - k_row_tile;
#else
        const uint32_t q_start_tile = 0;
        const uint32_t num_q_tiles_to_read = Ht;
#endif

#ifdef USE_ATTN_MASK
        const uint32_t mask_offset = k_row_tile;
#endif

        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
#ifdef USE_PRECOMPUTED_U_SCALER
            const uint32_t u_scaler_head_offset = (batch_idx * q_heads + first_q_head_idx + q_head_idx) * Ht;
#endif

            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_read; ++q_idx) {
                const uint32_t h = q_start_tile + q_idx;

                const uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_tiles_by_row(cb_query, query_address_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef USE_ATTN_MASK
                read_one_tile(cb_attn_mask, mask_address_generator, mask_offset + h * Ht);
#endif

                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_idx,
                    num_of_interm_tiles,
                    tile_bytes,
                    num_of_interm_tiles);

                read_tiles_by_row(cb_grad_output, grad_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef USE_PRECOMPUTED_U_SCALER
                read_one_tile(cb_u_scalar_row, u_scaler_address_generator, u_scaler_head_offset + h);
#else
                read_tiles_by_row(cb_attn_output, attn_output_address_generator, q_start_idx, qWt, tile_bytes, qWt);
#endif
            }
            intermediates_offset += Ht * num_of_interm_tiles;
        }
    }
#endif
}
