// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#define FLOAT32_REDUCTION true

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"

/**
 * @brief Compute kernel for distributed SUM reduction
 *
 * @details Performs two-phase reduction:
 * Phase 1: Local partial reduction - reduce across width (row-wise SUM)
 * Phase 2: Global reduction - reduce partial results from all cores
 *
 * @note Based on LayerNorm compute kernel but simplified for SUM reduction only
 */
namespace NAMESPACE {
void MAIN {
    // ============================================================================
    // Compile-time arguments
    // ============================================================================
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);  // num_cores_x
    constexpr uint32_t block_ht = get_compile_time_arg_val(1);    // Height per core
    constexpr uint32_t block_wt = get_compile_time_arg_val(2);    // Width per core

    static_assert(num_blocks > 1, "Need at least 2 cores for reduction");

    // ============================================================================
    // Runtime arguments
    // ============================================================================
    const uint32_t num_reduce_tiles_per_row = get_arg_val<uint32_t>(0);  // block_wt
    const uint32_t num_rows_per_worker = get_arg_val<uint32_t>(1);       // For final gather

    // ============================================================================
    // CB definitions
    // ============================================================================
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;       // Input data
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;    // Scaler (1/W for width reduction)
    constexpr uint32_t cb_partial = tt::CBIndex::c_2;   // Local partial results
    constexpr uint32_t cb_reduced = tt::CBIndex::c_3;   // Global reduced results
    constexpr uint32_t cb_external = tt::CBIndex::c_4;  // External (remote) partial data
    constexpr uint32_t cb_out = tt::CBIndex::c_5;       // Output (dataflow gathers directly here)

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    binary_op_init_common(cb_in0, cb_in0, cb_out);

    // ============================================================================
    // Phase 1: Local Partial Reduction - Reduce across width (row-wise SUM)
    // ============================================================================
    // For each row of tiles (height), reduce across width
    // Input: cb_in0 [block_ht x block_wt tiles]
    // Output: cb_partial [block_ht tiles] - one reduced tile per row

    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in0, cb_scaler, cb_partial);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_partial, block_ht);

    for (uint32_t row = 0; row < block_ht; ++row) {
        tile_regs_acquire();

        // Reduce across width for this row
        for (uint32_t col = 0; col < num_reduce_tiles_per_row; ++col) {
            uint32_t tile_idx = row * block_wt + col;
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in0, cb_scaler, tile_idx, scaler0, dst0);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_partial);
        tile_regs_release();
    }

    reduce_uninit();
    cb_push_back(cb_partial, block_ht);

    // ============================================================================
    // Phase 2: Global Reduction - Reduce partial results from all cores
    // ============================================================================
    // All cores participate in reading remote partials via dataflow kernel
    // Then perform global reduction across cores' partial results
    // Input: cb_external [num_rows_per_worker * num_blocks tiles]
    // Output: cb_reduced [num_rows_per_worker tiles]

    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_external, cb_scaler, cb_reduced);
    cb_reserve_back(cb_reduced, num_rows_per_worker);

    for (uint32_t row = 0; row < num_rows_per_worker; ++row) {
        cb_wait_front(cb_scaler, 1);
        tile_regs_acquire();

        // Reduce across all cores' partials for this row
        for (uint32_t core = 0; core < num_blocks; ++core) {
            cb_wait_front(cb_external, 1);
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_external, cb_scaler, 0, scaler0, dst0);
            cb_pop_front(cb_external, 1);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_reduced);
        tile_regs_release();
    }

    reduce_uninit();
    cb_push_back(cb_reduced, num_rows_per_worker);

    // ============================================================================
    // Phase 3: Done! Dataflow gathers results directly to cb_out
    // ============================================================================
    // Dataflow kernel (sender) will gather all cores' reduced results
    // directly to cb_out, so compute kernel is done after Phase 2.
}

}  // namespace NAMESPACE
