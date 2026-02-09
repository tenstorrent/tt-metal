// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/reduce_custom.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(0);          // Q input CB
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(1);          // K input CB (will be transposed)
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(2);          // V input CB
    constexpr uint32_t cb_qk_out = get_compile_time_arg_val(3);        // QK output CB
    constexpr uint32_t cb_max_out = get_compile_time_arg_val(4);       // Max output CB (from reduce)
    constexpr uint32_t cb_scale_in = get_compile_time_arg_val(5);      // Identity scale CB (for reduce)
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);           // Final output CB (QKV)
    constexpr uint32_t q_chunk_tiles = get_compile_time_arg_val(7);    // Q tiles (rows)
    constexpr uint32_t k_chunk_tiles = get_compile_time_arg_val(8);    // K tiles (cols after transpose)
    constexpr uint32_t inner_dim_tiles = get_compile_time_arg_val(9);  // Inner dimension for matmul

    // ====== Phase 1: QK Matmul only (for debugging) ======
    mm_init(cb_q_in, cb_k_in, cb_out, /*transpose=*/1);

    cb_push_back(cb_q_in, q_chunk_tiles * inner_dim_tiles);
    cb_push_back(cb_k_in, k_chunk_tiles * inner_dim_tiles);
    cb_reserve_back(cb_out, q_chunk_tiles * k_chunk_tiles);

    for (uint32_t q_tile = 0; q_tile < q_chunk_tiles; ++q_tile) {
        for (uint32_t k_tile = 0; k_tile < k_chunk_tiles; ++k_tile) {
            tile_regs_acquire();

            for (uint32_t inner = 0; inner < inner_dim_tiles; ++inner) {
                uint32_t q_idx = q_tile * inner_dim_tiles + inner;
                uint32_t k_idx = k_tile * inner_dim_tiles + inner;
                matmul_tiles(cb_q_in, cb_k_in, q_idx, k_idx, 0);
            }

            tile_regs_commit();
            tile_regs_wait();

            uint32_t out_idx = q_tile * k_chunk_tiles + k_tile;
            pack_tile(0, cb_out, out_idx);

            tile_regs_release();
        }
    }

    cb_push_back(cb_out, q_chunk_tiles * k_chunk_tiles);
}
