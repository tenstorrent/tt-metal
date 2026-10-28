// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t u_scaler_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices
    constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_7;    // Matmul row reduce tile
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_13;       // Output: grad_Q
    constexpr uint32_t cb_u_scaler_output = tt::CBIndex::c_14;  // Output: u_scaler for KV kernel

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // query width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(1);   // sequence length in tiles
    [[maybe_unused]] constexpr uint32_t q_heads = get_compile_time_arg_val(2);          // number of query heads
    [[maybe_unused]] constexpr uint32_t heads_per_group = get_compile_time_arg_val(3);  // Q heads per K/V head
    [[maybe_unused]] constexpr uint32_t heads_per_item = get_compile_time_arg_val(4);   // Q heads per work item

    // Generate helper tiles once at the start
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);  // tile for matmul row reduce

#ifdef CAUSAL_MASK
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);
    const uint32_t u_scaler_tile_bytes = get_tile_size(cb_u_scaler_output);

    // TensorAccessor definitions
    constexpr auto grad_query_args = TensorAccessorArgs<5>();
    constexpr auto u_scaler_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr);
    const auto u_scaler_addr_generator = TensorAccessor(u_scaler_args, u_scaler_output_addr);

    auto write_u_scaler_for_row = [&](const uint32_t global_row_idx) {
        // u_scaler tensor: one tile per query row, indexed by global_row_idx
        cb_wait_front(cb_u_scaler_output, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_u_scaler_output);
        noc_async_write_page(global_row_idx, u_scaler_addr_generator, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_u_scaler_output, onetile);
    };

#ifdef BALANCED_PARALLELISM
    constexpr uint32_t pairs_per_seq = Ht / 2;

    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

#ifdef GROUPED_HEADS
        // Grouped item = (batch, kv_head, head sub-group, q_row). Compute emits the heads_per_item
        // u tiles at item start and the heads_per_item dQ rows at item end -> write in that order.
        constexpr uint32_t items_per_group = heads_per_group / heads_per_item;
        constexpr uint32_t num_of_groups = q_heads / heads_per_group;
        auto write_item = [&](const uint32_t global_item_idx) {
            const uint32_t q_row_tile = global_item_idx % Ht;
            const uint32_t item_seq_idx = global_item_idx / Ht;
            const uint32_t kv_seq_idx = item_seq_idx / items_per_group;
            const uint32_t sub_group_idx = item_seq_idx % items_per_group;
            const uint32_t batch_idx = kv_seq_idx / num_of_groups;
            const uint32_t kv_group_idx = kv_seq_idx % num_of_groups;
            const uint32_t first_q_head = kv_group_idx * heads_per_group + sub_group_idx * heads_per_item;
            for (uint32_t j = 0; j < heads_per_item; ++j) {
                write_u_scaler_for_row((batch_idx * q_heads + first_q_head + j) * Ht + q_row_tile);
            }
            for (uint32_t j = 0; j < heads_per_item; ++j) {
                const uint32_t global_q_row = (batch_idx * q_heads + first_q_head + j) * Ht + q_row_tile;
                write_tiles_by_row(cb_grad_query, grad_query_addr_generator, global_q_row * qWt, qWt, tile_bytes, qWt);
            }
        };
        write_item(heavy_global_row);
        write_item(light_global_row);
#else
        // Write heavy row grad_query + u_scaler
        const uint32_t heavy_start_idx = heavy_global_row * qWt;
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, heavy_start_idx, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(heavy_global_row);

        // Write light row grad_query + u_scaler
        const uint32_t light_start_idx = light_global_row * qWt;
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, light_start_idx, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(light_global_row);
#endif
    }
#else
    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t q_start_idx = r * qWt;

        // Write grad_query row
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);

        // Write u_scaler tile for this row
        write_u_scaler_for_row(r);
    }
#endif
}
