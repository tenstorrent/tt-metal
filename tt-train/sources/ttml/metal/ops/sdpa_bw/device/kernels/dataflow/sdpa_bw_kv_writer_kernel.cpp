// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices for gradients
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_15;    // Output: grad_K
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_16;  // Output: grad_V

    // Get compile-time arguments
    constexpr uint32_t kWt = get_compile_time_arg_val(0);              // key width in tiles (grad_K)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);              // value width in tiles (grad_V)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // heads per group
    [[maybe_unused]] constexpr uint32_t kv_rows_per_item =
        get_compile_time_arg_val(5);  // K/V tile-rows per work item (>1 only with GROUPED_KV_ROWS)

#ifdef CAUSAL_MASK
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes_k = get_tile_size(cb_grad_key);
    const uint32_t tile_bytes_v = get_tile_size(cb_grad_value);

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_key_args = TensorAccessorArgs<6>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators for output gradients
    const auto grad_key_addr_generator = TensorAccessor(grad_key_args, grad_key_addr, tile_bytes_k);
    const auto grad_value_addr_generator = TensorAccessor(grad_value_args, grad_value_addr, tile_bytes_v);

    const uint32_t num_of_groups = q_heads / heads_per_group;

#ifdef BALANCED_PARALLELISM
#ifdef GROUPED_KV_ROWS
    constexpr uint32_t items_per_seq = Ht / kv_rows_per_item;
#else
    constexpr uint32_t items_per_seq = Ht;
#endif
    constexpr uint32_t pairs_per_seq = items_per_seq / 2;

    auto write_row = [&](const uint32_t global_row_idx) {
        const uint32_t batch_idx = global_row_idx / (num_of_groups * Ht);
        const uint32_t s_tile_idx = global_row_idx % Ht;
        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;

        const uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * vWt;
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, grad_v_row_base_tiles, vWt, tile_bytes_v, vWt);

        const uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, grad_k_row_base_tiles, kWt, tile_bytes_k, kWt);
    };

    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_item_in_seq = pair_in_seq;
        const uint32_t heavy_item_in_seq = items_per_seq - 1 - pair_in_seq;

        // Compute emits dV_r, dK_r for each of the item's kv_rows_per_item consecutive rows in order.
        auto write_item = [&](const uint32_t item_in_seq) {
            const uint32_t first_row = seq_idx * Ht + item_in_seq * kv_rows_per_item;
            for (uint32_t r = 0; r < kv_rows_per_item; ++r) {
                write_row(first_row + r);
            }
        };
        write_item(heavy_item_in_seq);
        write_item(light_item_in_seq);
    }
#else
    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // Convert global row index to tensor coordinates
        const uint32_t batch_idx = r / (num_of_groups * Ht);
        const uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)

        const uint32_t group_idx = (r / Ht) % num_of_groups;  // which group of K and V we are processing

        // -------- Grad Value: same shape as Value (B, vNH, S, vEmbd) --------
        const uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * vWt;
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, grad_v_row_base_tiles, vWt, tile_bytes_v, vWt);

        // -------- Grad Key: same shape as Key (B, kNH, S, kEmbd) --------
        const uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, grad_k_row_base_tiles, kWt, tile_bytes_k, kWt);
    }
#endif
}
