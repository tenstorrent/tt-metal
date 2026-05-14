// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Numerically stable row-wise softmax: processes Mt tile-rows, each Wt tiles wide.
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//
// CB layout:
//   cb_in (c_0):     input tiles, sized for Wt tiles (loaded all at once per row)
//   cb_scaler (c_1): tile filled with 1.0 for reduce scaling (1 tile, persistent)
//   cb_exps (c_2):   exp results, sized for Wt tiles
//   cb_recip (c_3):  1/sum result (1 tile)
//   cb_max (c_4):    max per row (1 tile)
//   cb_sub (c_5):    x - max intermediate (Wt tiles)
//   cb_out (c_16):   output tiles

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Wt = get_compile_time_arg_val(1);

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_scaler = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_exps = tt::CBIndex::c_2;
    constexpr tt::CBIndex cb_recip = tt::CBIndex::c_3;
    constexpr tt::CBIndex cb_max = tt::CBIndex::c_4;
    constexpr tt::CBIndex cb_sub = tt::CBIndex::c_5;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_in, cb_scaler, cb_exps);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        cb_wait_front(cb_in, Wt);

        // Step 1: Find row-wise max
        cb_wait_front(cb_scaler, 1);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_max);
        tile_regs_acquire();
        for (uint32_t n = 0; n < Wt; n++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, n, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_max, 1);
        pack_tile(0, cb_max);
        cb_push_back(cb_max, 1);
        tile_regs_release();
        reduce_uninit();

        // Step 2: x - max (broadcast column)
        cb_wait_front(cb_max, 1);
        sub_bcast_cols_init_short(cb_in, cb_max);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_in, cb_max, n, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_sub, 1);
            pack_tile(0, cb_sub);
            cb_push_back(cb_sub, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_in, Wt);
        cb_pop_front(cb_max, 1);

        // Step 3: exp(x - max)
        cb_wait_front(cb_sub, Wt);
        init_sfpu(cb_sub, cb_exps);
        exp_tile_init<true>();
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            copy_tile(cb_sub, n, 0);
            exp_tile<true>(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_exps, 1);
            pack_tile(0, cb_exps);
            cb_push_back(cb_exps, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_sub, Wt);

        // Step 4: reduce sum across exp tiles (row-wise) -> reciprocal
        cb_wait_front(cb_exps, Wt);

        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, cb_recip);
        tile_regs_acquire();
        for (uint32_t n = 0; n < Wt; n++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_exps, cb_scaler, n, 0, 0);
        }
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_recip, 1);
        pack_tile(0, cb_recip);
        cb_push_back(cb_recip, 1);
        tile_regs_release();
        reduce_uninit();

        // Step 5: multiply each exp tile by 1/sum (broadcast column)
        cb_wait_front(cb_recip, 1);
        mul_bcast_cols_init_short(cb_exps, cb_recip);
        for (uint32_t n = 0; n < Wt; n++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_exps, cb_recip, n, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_exps, Wt);
        cb_pop_front(cb_recip, 1);
    }
}
