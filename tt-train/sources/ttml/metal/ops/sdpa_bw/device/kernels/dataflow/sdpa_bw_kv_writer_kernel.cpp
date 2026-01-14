// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_17;    // Output: grad_K
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_18;  // Output: grad_V

    // Get compile-time arguments
    constexpr uint32_t kWt = get_compile_time_arg_val(0);              // key/value width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(1);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(2);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(3);  // heads per group

    const uint32_t tile_bytes = get_tile_size(cb_grad_key);

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_key_args = TensorAccessorArgs<4>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators for output gradients
    const auto grad_key_addr_generator = TensorAccessor(grad_key_args, grad_key_addr, tile_bytes);
    const auto grad_value_addr_generator = TensorAccessor(grad_value_args, grad_value_addr, tile_bytes);

    const uint32_t num_of_groups = q_heads / heads_per_group;

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // Convert global row index to tensor coordinates
        const uint32_t batch_idx = r / (num_of_groups * Ht);
        const uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)

        const uint32_t group_idx = (r / Ht) % num_of_groups;  // which group of K and V we are processing

        // -------- Grad Value: same shape as Value (B, vNH, S, vEmbd) --------
        const uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, grad_v_row_base_tiles, kWt, tile_bytes, kWt);

        // -------- Grad Key: same shape as Key (B, kNH, S, kEmbd) --------
        const uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, grad_k_row_base_tiles, kWt, tile_bytes, kWt);
    }
}
