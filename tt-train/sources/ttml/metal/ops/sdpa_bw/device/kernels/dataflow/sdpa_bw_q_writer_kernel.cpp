// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices
    constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_7;  // Matmul row reduce tile
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_13;     // Output: grad_Q

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // query width in tiles

    // Generate helper tiles once at the start
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);  // tile for matmul row reduce

#ifdef CAUSAL_MASK
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);

    // TensorAccessor definitions
    constexpr auto grad_query_args = TensorAccessorArgs<1>();

    // Create TensorAccessor generator for output gradient
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t q_start_idx = r * qWt;

        // Write grad_query row on same position as read(same output shape)
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);
    }
}
