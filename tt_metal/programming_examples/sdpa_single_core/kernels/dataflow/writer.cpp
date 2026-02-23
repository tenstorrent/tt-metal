// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(6);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Generate column identity tile for matmul_reduce normalization.
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // Normalized output streams through a dedicated 1-tile CB (decoupled from ping-pong out CBs).
    constexpr uint32_t cb_normalized_out = tt::CBIndex::c_9;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;

    uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr auto out_accessor_args = TensorAccessorArgs<7>();
    const auto out_accessor = TensorAccessor(out_accessor_args, out_addr, get_tile_size(cb_normalized_out));

    constexpr uint32_t subblock_h = 1;  // Must match compute kernel's subblock_h
    constexpr uint32_t row_tiles = subblock_h * head_dim_t;
    constexpr uint32_t rows_per_chunk = Sq_chunk_t / subblock_h;

    for (uint32_t q = 0; q < num_q_chunks; q++) {
        uint32_t tile_base = q * out_chunk_tiles;
        for (uint32_t row = 0; row < rows_per_chunk; row++) {
            // Issue all tile writes for one row, then barrier once
            for (uint32_t t = 0; t < row_tiles; t++) {
                cb_wait_front(cb_normalized_out, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_normalized_out);
                noc_async_write_tile(tile_base + row * row_tiles + t, out_accessor, l1_read_addr);
                cb_pop_front(cb_normalized_out, 1);
            }
            noc_async_write_barrier();
        }
    }
}
