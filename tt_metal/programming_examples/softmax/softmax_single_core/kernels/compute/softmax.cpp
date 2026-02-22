// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define EXP_APPROX false

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/compute_kernel_hw_startup.h"

// Two-pass row-wise softmax compute kernel.
//
// ── Pass 1  Row-wise max → cb_max  ────────────────────────────────────────
//   Consume Nt input tiles one at a time.  Each tile is reduced into DST[0]
//   using reduce_tile<MAX, REDUCE_ROW>.  After the loop, DST[0][r, 0] holds
//   the maximum value for row r (other columns = 0).
//   The result is packed to cb_max (1 tile: col 0 = per-row maxima).
//
// ── Pass 2  exp(x − max) → cb_max  ───────────────────────────────────────
//   For each input tile, sub_tiles_bcast_cols subtracts the row max (cb_max[0],
//   broadcast across all 32 columns); exp_tile then computes exp(x − max).
//   Each resulting tile is pushed to the BACK of cb_max, so by the end of the
//   loop cb_max contains [row-max | exp[0] | … | exp[Nt-1]].
//   The row-max tile is then popped, leaving Nt exp tiles in cb_max.
//
// ── Pass 3  Row-wise sum of exp tiles → 1/sum in cb_sum  ─────────────────
//   All Nt exp tiles in cb_max are reduced with reduce_tile<SUM, REDUCE_ROW>
//   (using explicit tile indices, so nothing is popped from cb_max yet).
//   recip_tile converts the per-row sums to 1/sum, which is packed to cb_sum.
//
// ── Pass 4  Normalise → cb_out  ───────────────────────────────────────────
//   Each exp tile is popped from cb_max, multiplied element-wise by 1/sum
//   (col 0 of cb_sum, broadcast across columns), and packed to cb_out.
//
// Compile-time arguments:
//   0: Mt — Number of tile rows.
//   1: Nt — Number of tile columns.
//
// Circular buffers:
//   c_0  (cb_in):     Input tiles (double-buffered, 2 tiles).
//   c_1  (cb_scaler): Constant 1.0 tile for reduce (1 tile, never popped).
//   c_2  (cb_max):    Capacity Nt+1: first the row-max tile, then Nt exp tiles.
//   c_3  (cb_sum):    1/sum tile — 1 tile per mt, produced by pass 3.
//   c_16 (cb_out):    Softmax output (Mt×Nt tiles).

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Nt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_in     = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_max    = tt::CBIndex::c_2;
    constexpr uint32_t cb_sum    = tt::CBIndex::c_3;
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;

    constexpr uint32_t dst_max = 0;
    constexpr uint32_t dst_sum = 1;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        // ── Pass 1: row-wise max → cb_max (1 tile) ───────────────────────
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_max);
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_in, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, dst_max);
            cb_pop_front(cb_in, 1);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();

        // Pack the row-max tile to cb_max.
        // Layout: col 0 = per-row maxima, cols 1-31 = 0.
        cb_reserve_back(cb_max, 1);
        pack_tile(dst_max, cb_max);
        cb_push_back(cb_max, 1);
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
            pack_tile(dst_max, cb_max);   // push exp tile to back of cb_max
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
        reduce_uninit();
        recip_tile_init();
        recip_tile(dst_sum);   // DST[dst_sum] = 1 / per-row-sum
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_sum, 1);
        pack_tile(dst_sum, cb_sum);   // col 0 = 1/sum_r, broadcast target for pass 4
        cb_push_back(cb_sum, 1);
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
