// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// RoPE compute kernel
// Computes: output = (input * cos) + (rotate_half(input) * sin)
// where rotate_half(input) = input @ trans_mat

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);  // head_dim in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // n_heads in tiles

    // Initialize matmul and binary ops (done once)
    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);

    constexpr uint32_t onetile = 1;

    // ========================================================================
    // Signal sharded CBs are ready (trans_mat, sin, cos)
    // For sharded tensors: reserve + push makes data "visible" to compute
    // ========================================================================

    // Trans_mat: 1 tile, reused for all heads
    cb_reserve_back(trans_mat_cb, onetile);
    cb_push_back(trans_mat_cb, onetile);
    cb_wait_front(trans_mat_cb, onetile);

    // Sin/Cos: Wt tiles each, broadcast across all heads
    cb_reserve_back(sin_cb, Wt);
    cb_reserve_back(cos_cb, Wt);
    cb_push_back(sin_cb, Wt);
    cb_push_back(cos_cb, Wt);

    // ========================================================================
    // Main loop: process each head tile row
    // ========================================================================
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Reserve intermediate and output buffers
        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(sin_interm_cb, Wt);
        cb_reserve_back(cos_interm_cb, Wt);
        cb_reserve_back(out_cb, Wt);

        // Signal input row is ready (sharded tensor)
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // ====================================================================
        // Step 1: rotated = input @ trans_mat (matmul for rotate_half)
        // ====================================================================
        mm_init_short(in_cb, trans_mat_cb);
        acquire_dst();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        release_dst();
        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        // ====================================================================
        // Step 2: sin_interm = rotated * sin (broadcast multiply)
        // ====================================================================
        mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
        acquire_dst();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb, sin_cb, j, j, j);
            pack_tile(j, sin_interm_cb, j);
        }
        release_dst();
        cb_push_back(sin_interm_cb, Wt);
        cb_pop_front(rotated_in_interm_cb, Wt);

        // ====================================================================
        // Step 3: cos_interm = input * cos (broadcast multiply)
        // ====================================================================
        acquire_dst();
        for (uint32_t j = 0; j < Wt; ++j) {
            mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j, j, j);
            pack_tile(j, cos_interm_cb, j);
        }
        release_dst();
        cb_push_back(cos_interm_cb, Wt);
        cb_pop_front(in_cb, Wt);

        // ====================================================================
        // Step 4: output = cos_interm + sin_interm (add)
        // ====================================================================
        cb_wait_front(sin_interm_cb, Wt);
        cb_wait_front(cos_interm_cb, Wt);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        acquire_dst();
        for (uint32_t j = 0; j < Wt; ++j) {
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            pack_tile(j, out_cb, j);
        }
        release_dst();
        cb_push_back(out_cb, Wt);
        cb_pop_front(sin_interm_cb, Wt);
        cb_pop_front(cos_interm_cb, Wt);
    }

    // ========================================================================
    // Cleanup: pop shared tensors
    // ========================================================================
    cb_pop_front(sin_cb, Wt);
    cb_pop_front(cos_cb, Wt);
    cb_pop_front(trans_mat_cb, onetile);
}
}  // namespace NAMESPACE
