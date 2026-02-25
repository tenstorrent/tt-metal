// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"

using std::uint32_t;

/**
 * @brief Compute kernel: row-wise maximum reduction (multi-core).
 *
 * Identical in structure to the single-core version, but mt_count is a
 * runtime argument instead of compile-time so that each core can receive
 * a different (possibly unequal) share of the Mt row groups.
 *
 * Compile-time arguments:
 *   0: Nt       - Number of input tile columns (the reduction dimension).
 *                 Same for every core; kept compile-time for loop unrolling.
 *
 * Runtime arguments:
 *   0: mt_count - Number of tile-row groups assigned to this core.
 *
 * Circular buffers:
 *   c_0  (cb_in):     Input tiles, double-buffered.
 *   c_1  (cb_scaler): Scaler tile (all 1.0), never popped.
 *   c_16 (cb_out):    Output tiles, one per row group.
 */
void kernel_main() {
    // TODO: implement multi-core compute kernel.
    //
    // Steps:
    //   1. Read mt_count from runtime arg 0.
    //   2. Read Nt from compile-time arg 0.
    //   3. Call compute_kernel_hw_startup(cb_in, cb_scaler, cb_out).
    //   4. Call reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out).
    //   5. cb_wait_front(cb_scaler, 1).
    //   6. For each mt in [0, mt_count):
    //        tile_regs_acquire();
    //        For each nt in [0, Nt):
    //          cb_wait_front(cb_in, 1);
    //          reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
    //          cb_pop_front(cb_in, 1);
    //        tile_regs_commit();
    //        tile_regs_wait();
    //        cb_reserve_back(cb_out, 1);
    //        pack_tile(0, cb_out);
    //        cb_push_back(cb_out, 1);
    //        tile_regs_release();

    uint32_t mt_count = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;  

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t mt = 0; mt < mt_count; mt++) {
        tile_regs_acquire();
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_in, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
            cb_pop_front(cb_in, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
