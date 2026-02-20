// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "hostdevcommon/dprint_common.h"

// These macros provide default template arguments for reduce_init/reduce_tile.
// We also pass them explicitly as template parameters for clarity.
#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/debug/dprint.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint_tile.h"

using std::uint32_t;

/**
 * @brief Compute kernel: row-wise maximum reduction.
 *
 * For an input matrix of shape M×N (stored as Mt×Nt tiles), this kernel computes
 * the maximum value across all N columns for each of the M rows.
 * The result is an Mt-tile output where each tile holds 32 row maxima in column 0
 * (all other positions are zero).
 *
 * Compile-time arguments:
 *   - Mt: Number of input/output tile rows.
 *   - Nt: Number of input tile columns (the dimension being reduced).
 *
 * Circular buffers:
 *   - c_0  (cb_in):     Input tiles, Nt tiles capacity (one full row of tiles at a time).
 *   - c_1  (cb_scaler): Scaler tile with all 1.0 values (one tile, never popped).
 *   - c_16 (cb_out):    Output tiles, one reduced tile per row group.
 */
void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Nt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize HW (UNPACK, MATH, PACK engines) and DST semaphore state.
    // Must be called once before any other compute API calls.
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // Configure the math, unpack, and pack engines for REDUCE_ROW MAX.
    // Called once before the outer loop; stays active for all Mt iterations.
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out);

    // The scaler tile is sent once by the reader and is never popped.
    // Wait for it before starting the main computation loop.
    cb_wait_front(cb_scaler, 1);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        // tile_regs_acquire() makes MATH wait for DST to be available (previous pack released it),
        // and makes PACK wait for MATH to commit. This properly serializes MATH and PACK.
        tile_regs_acquire();

        // Process all Nt column tiles that belong to this row group.
        // Each reduce_tile call compares the incoming per-row values against the running
        // maximum already in DST[0], keeping the larger value.
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_in, 1);
            // Input tile index 0 (front of CB); scaler tile index 0; result in DST[0].
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
            cb_pop_front(cb_in, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Pack DST[0] into the output CB and release DST for the next iteration.
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        DPRINT_PACK({
            DPRINT << "DST[0] col0 (mt=" << mt << "):" << ENDL();
            DPRINT << TSLICE(cb_out, 0, SliceRange::h0_32_w0()) << ENDL();
            // DPRINT << "DST[0] row0 (mt=" << mt << "):" << ENDL();
            // DPRINT << TSLICE(cb_out, 0, SliceRange::h0_w0_32()) << ENDL();
        });
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}