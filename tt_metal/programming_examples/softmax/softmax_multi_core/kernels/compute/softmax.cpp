// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define EXP_APPROX false

#define REDUCE_OP PoolType::MAX
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/compute_kernel_hw_startup.h"

// Four-pass row-wise softmax compute kernel (multi-core).
//
// Identical to the single-core version; the only difference is that the number
// of tile-row groups (mt_count) is a runtime argument so each core can be
// assigned a different share by the host.  Nt stays compile-time to allow inner
// loop bounds to be constant-folded.
//
// Compile-time arguments:
//   0: Nt       — Number of tile columns (same for every core).
//
// Runtime arguments:
//   0: mt_count — Number of tile-row groups assigned to this core.
//
// Circular buffers:
//   c_0  (cb_in):     Input tiles (double-buffered, 2 tiles).
//   c_1  (cb_scaler): Constant 1.0 tile for reduce (1 tile, never popped).
//   c_2  (cb_max):    Capacity Nt+1: first the row-max tile, then Nt exp tiles.
//   c_3  (cb_sum):    1/sum tile — 1 tile per mt, produced by pass 3.
//   c_16 (cb_out):    Softmax output (Nt tiles capacity, one row at a time).

void kernel_main() {
    const uint32_t Nt       = get_compile_time_arg_val(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in     = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_max    = tt::CBIndex::c_2;
    constexpr uint32_t cb_sum    = tt::CBIndex::c_3;
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;

    constexpr uint32_t dst_max = 0;
    constexpr uint32_t dst_sum = 1;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t mt = 0; mt < mt_count; mt++) {
        // ── Pass 1: row-wise max → cb_max (1 tile) ───────────────────────
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_max);
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_in, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, dst_max);
            cb_pop_front(cb_in, 1);
        }
        tile_regs_commit();
        tile_regs_wait();

        // Pack the row-max tile to cb_max.
        // Layout: col 0 = per-row maxima, cols 1-31 = 0.
        cb_reserve_back(cb_max, 1);
        pack_tile(dst_max, cb_max);
        cb_push_back(cb_max, 1);
        reduce_uninit();
        tile_regs_release();

        // ── Pass 2: exp(x − max) → cb_max (Nt tiles after row-max) ──────
        // sub_tiles_bcast_cols always reads cb_max at tile index 0, which is
        // the row-max tile pushed above.  Each exp result is pushed to the
        // back of cb_max without disturbing index 0.
        sub_bcast_cols_init_short(cb_in, cb_max);
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_in, 1);
            tile_regs_acquire();
            cb_reserve_back(cb_max, 1);

            sub_tiles_bcast_cols(cb_in, cb_max, 0, 0, dst_max);  // DST[0] = x − max
            exp_tile_init<EXP_APPROX>();
            exp_tile<EXP_APPROX>(dst_max);                        // DST[0] = exp(x − max)

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_max, cb_max);  // push exp tile to back of cb_max
            cb_push_back(cb_max, 1);
            tile_regs_release();
            cb_pop_front(cb_in, 1);
        }

        // Pop the row-max tile.  cb_max now holds exactly Nt exp(x − max) tiles.
        cb_pop_front(cb_max, 1);

        // ── Pass 3: row-wise sum → 1/sum in cb_sum ───────────────────────
        // Wait for all Nt exp tiles, then reduce with explicit tile indices so
        // nothing is popped.  recip_tile converts the per-row sums to 1/sum.
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_max, cb_scaler, cb_sum);
        cb_wait_front(cb_max, Nt);
        for (uint32_t nt = 0; nt < Nt; nt++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_max, cb_scaler, nt, 0, dst_sum);
        }
        recip_tile_init();
        recip_tile(dst_sum);  // DST[dst_sum] = 1 / per-row-sum
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_sum, 1);
        pack_tile(dst_sum, cb_sum);  // col 0 = 1/sum_r, broadcast target for pass 4
        cb_push_back(cb_sum, 1);
        reduce_uninit();
        tile_regs_release();

        // ── Pass 4: exp(x − max) × (1/sum) → cb_out ─────────────────────
        // Iterate through the Nt exp tiles still in cb_max, multiply each
        // element by the corresponding per-row 1/sum (col 0 of cb_sum,
        // broadcast across columns), and emit to cb_out.
        mul_bcast_cols_init_short(cb_max, cb_sum);
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_max, 1);
            tile_regs_acquire();
            cb_reserve_back(cb_out, 1);

            mul_tiles_bcast_cols(cb_max, cb_sum, 0, 0, dst_max);  // DST[0] = exp * (1/sum)

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_max, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
            cb_pop_front(cb_max, 1);
        }

        cb_pop_front(cb_sum, 1);
    }
}
