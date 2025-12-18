// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t grad_query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices for gradients
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_15;  // Output: grad_Q

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // query width in tiles

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);

    // TensorAccessor definitions
    constexpr auto grad_query_args = TensorAccessorArgs<1>();

    // Create TensorAccessor generator for output gradient
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        uint32_t global_row_idx = r;
        uint32_t q_start_idx = global_row_idx * qWt;

        // Write grad_query row on same position as read(same output shape)
        cb_wait_front(cb_grad_query, qWt);
        uint32_t l1_read_addr = get_read_ptr(cb_grad_query);
        for (uint32_t tile_idx = 0; tile_idx < qWt; ++tile_idx) {
            noc_async_write_tile(q_start_idx + tile_idx, grad_query_addr_generator, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_grad_query, qWt);
    }
}
