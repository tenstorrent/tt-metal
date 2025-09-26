// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>
#include <debug/dprint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t grad_query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t grad_key_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t grad_value_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices for gradients
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_9;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_10;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_11;

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);         // query width in tiles
    constexpr uint32_t kWt = get_compile_time_arg_val(1);         // key/value width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(2);          // sequence length in tiles
    constexpr uint32_t block_size = get_compile_time_arg_val(3);  // block size
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);     // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5); // heads per group

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);
    const DataFormat data_format = get_dataformat(cb_grad_query);

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_query_args = TensorAccessorArgs<6>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators for output gradients
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr, tile_bytes);
    const auto grad_key_addr_generator = TensorAccessor(grad_key_args, grad_key_addr, tile_bytes);
    const auto grad_value_addr_generator = TensorAccessor(grad_value_args, grad_value_addr, tile_bytes);

    const uint32_t num_of_groups = q_heads / heads_per_group;
    
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // Convert global row index to tensor coordinates
        uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)
        uint32_t q_head_idx = (r / Ht) % q_heads;
        uint32_t batch_idx = r / (Ht * q_heads);

        // -------- Grad Query: same shape as Query (B, qNH, S, qEmbd) --------
        // Row base for (batch_idx, q_head_idx, s_tile): ((b * q_heads + h) * Ht + s_tile) * qWt
        uint32_t grad_q_row_base_tiles = ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * qWt;

        cb_wait_front(cb_grad_query, qWt);
        uint32_t l1_grad_q_read_addr = get_read_ptr(cb_grad_query);
        for (uint32_t col = 0; col < qWt; ++col) {
            noc_async_write_tile(grad_q_row_base_tiles + col, grad_query_addr_generator, l1_grad_q_read_addr);
            l1_grad_q_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_grad_query, qWt);

        // -------- Grad Key: same shape as Key (B, kNH, S, kEmbd) --------
        uint32_t kv_group_idx = q_head_idx / heads_per_group;  // which KV group this Q head belongs to
        uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + kv_group_idx) * Ht + s_tile_idx) * kWt;

        cb_wait_front(cb_grad_key, kWt);
        uint32_t l1_grad_k_read_addr = get_read_ptr(cb_grad_key);
        for (uint32_t col = 0; col < kWt; ++col) {
            noc_async_write_tile(grad_k_row_base_tiles + col, grad_key_addr_generator, l1_grad_k_read_addr);
            l1_grad_k_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_grad_key, kWt);

        // -------- Grad Value: same shape as Value (B, vNH, S, vEmbd) --------
        uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + kv_group_idx) * Ht + s_tile_idx) * kWt;

        cb_wait_front(cb_grad_value, kWt);
        uint32_t l1_grad_v_read_addr = get_read_ptr(cb_grad_value);
        for (uint32_t col = 0; col < kWt; ++col) {
            noc_async_write_tile(grad_v_row_base_tiles + col, grad_value_addr_generator, l1_grad_v_read_addr);
            l1_grad_v_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_grad_value, kWt);
    }
}
