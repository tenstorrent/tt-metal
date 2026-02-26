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
    constexpr uint32_t subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t padded_k_tiles = get_compile_time_arg_val(8);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // Generate column identity tile for matmul_reduce normalization.
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // Generate a single -inf tile for padded-K masking.
    // Compute kernel L1-accumulates this onto padded tile positions in the row buffer.
    // Stays fronted (never popped) for reuse across all Q subblocks and rows.
    if constexpr (padded_k_tiles > 0) {
        constexpr uint32_t cb_mask_in = tt::CBIndex::c_7;
        const uint32_t tile_size_bytes = get_tile_size(cb_mask_in);

        cb_reserve_back(cb_mask_in, 1);
        auto* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_mask_in));
        for (uint32_t i = 0; i < tile_size_bytes / sizeof(uint32_t); i++) {
            ptr[i] = 0xFF80FF80;  // -inf in bfloat16
        }
        cb_push_back(cb_mask_in, 1);
    }

    // Normalized output streams through a double-buffered CB (decoupled from ping-pong out CBs).
    // Compute pushes head_dim_t tiles at a time; we wait for the full batch, issue all writes
    // back-to-back (pipelining NoC transactions), then pop.
    constexpr uint32_t cb_normalized_out = tt::CBIndex::c_9;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * head_dim_t;
    const uint32_t tile_size_bytes = get_tile_size(cb_normalized_out);

    uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr auto out_accessor_args = TensorAccessorArgs<9>();
    const auto out_accessor = TensorAccessor(out_accessor_args, out_addr, tile_size_bytes);

    // subblock_h from compile-time arg (index 7), must match compute kernel
    constexpr uint32_t row_tiles = subblock_h * head_dim_t;
    constexpr uint32_t rows_per_chunk = Sq_chunk_t / subblock_h;

    for (uint32_t q = 0; q < num_q_chunks; q++) {
        uint32_t tile_base = q * out_chunk_tiles;
        for (uint32_t row = 0; row < rows_per_chunk; row++) {
            // Wait for each compute push (head_dim_t tiles), issue writes back-to-back, pop batch.
            // With sbh>1, compute pushes head_dim_t tiles × subblock_h times per writer row.
            for (uint32_t s = 0; s < subblock_h; s++) {
                cb_wait_front(cb_normalized_out, head_dim_t);
                uint32_t l1_base = get_read_ptr(cb_normalized_out);
                for (uint32_t t = 0; t < head_dim_t; t++) {
                    noc_async_write_tile(
                        tile_base + row * row_tiles + s * head_dim_t + t, out_accessor, l1_base + t * tile_size_bytes);
                }
                cb_pop_front(cb_normalized_out, head_dim_t);
            }
            noc_async_write_barrier();
        }
    }
}
