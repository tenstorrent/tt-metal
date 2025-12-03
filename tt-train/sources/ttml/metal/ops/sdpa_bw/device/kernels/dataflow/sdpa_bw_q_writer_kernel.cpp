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
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices for gradients
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_18;          // Output: grad_Q
    constexpr uint32_t cb_sync_output_writer = tt::CBIndex::c_21;  // Used to sync with output writer kernel

    // TODO(vmelnykov): delete this after testing
    constexpr uint32_t cb_attention_weights =
        tt::CBIndex::c_13;  // Recomputed attention weights = softmax(QK^T / sqrt(Et))
    constexpr uint32_t cb_grad_attn_weights =
        tt::CBIndex::c_14;  // Gradient w.r.t. attention: dL/dP = dL/d(softmax(QK^T / sqrt(Et)))
    constexpr uint32_t cb_transpose_wh = tt::CBIndex::c_16;  // Transpose of key/value

    // [DEBUG]: Used for debug, should be removed later
    constexpr auto cb_masked_interm = tt::CBIndex::c_22;

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // query width in tiles
    constexpr uint32_t kWt = get_compile_time_arg_val(1);              // key/value width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t block_size = get_compile_time_arg_val(3);       // block size
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);  // heads per group

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);
    const DataFormat data_format = get_dataformat(cb_grad_query);

    // TensorAccessor definitions
    constexpr auto grad_query_args = TensorAccessorArgs<6>();

    // Create TensorAccessor generator for output gradient
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr, tile_bytes);

    const uint32_t num_of_groups = q_heads / heads_per_group;
    const uint32_t onetile = 1U;

    DPRINT << "Q_WRITER: num_rows_to_process=" << num_rows_to_process << ", start_row=" << start_row << ENDL();

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        uint32_t global_row_idx = r;
        uint32_t q_start_idx = global_row_idx * qWt;

        // DPRINT << "Q_WRITER: Writing row " << r << ", q_start_idx=" << q_start_idx << ENDL();
        // cb_wait_front(cb_masked_interm, 2U);
        // print_tile(cb_masked_interm, 0);
        // print_tile(cb_masked_interm, 1);

        // Write grad_query row on same position as read(same output shape)
        cb_wait_front(cb_grad_query, qWt);
        uint32_t l1_read_addr = get_read_ptr(cb_grad_query);
        for (uint32_t tile_idx = 0; tile_idx < qWt; ++tile_idx) {
            noc_async_write_tile(q_start_idx + tile_idx, grad_query_addr_generator, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_grad_query, qWt);

        // cb_wait_front(cb_attention_weights, onetile);
        // uint32_t l1_read_addr = get_read_ptr(cb_attention_weights);
        // noc_async_write_tile(q_start_idx, grad_query_addr_generator, l1_read_addr);
        // noc_async_write_barrier();
        // cb_pop_front(cb_attention_weights, onetile);

        // cb_wait_front(cb_transpose_wh, onetile);
        // print_tile(cb_transpose_wh, 0);

        // print attn weights tile
        // cb_wait_front(cb_sync_output_writer, onetile);
        // print_tile(cb_sync_output_writer, 0);
        // uint32_t l1_read_addr = get_read_ptr(cb_sync_output_writer);
        // noc_async_write_tile(q_start_idx, grad_query_addr_generator, l1_read_addr);
        // noc_async_write_barrier();
        // cb_pop_front(cb_sync_output_writer, onetile);
    }
}
