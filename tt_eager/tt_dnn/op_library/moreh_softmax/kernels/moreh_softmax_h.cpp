// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/reduce.h"

#include "debug_print.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_mask = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;

    binary_op_init_common(cb_in0, cb_bcast_scaler);

    cb_wait_front(cb_bcast_scaler, onetile); // comes from the reader
    cb_wait_front(cb_mask, onetile); // comes from the reader

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    constexpr int dst0 = 0;
    for(uint32_t n = 0; n < N; ++n) {

        // step 1, compute exp(x)
        for(uint32_t h = 0; h < Ht; ++h)
        {
            ACQ();
            cb_wait_front(cb_in0, onetile);
            copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
            copy_tile(cb_in0, 0, dst0);
            cb_pop_front(cb_in0, onetile);

            cb_reserve_back(cb_exps, onetile);
            exp_tile_init();
            exp_tile(dst0);

            if (h == Ht - 1) {
                constexpr int dst_mask = 1;
                copy_tile_init();
                copy_tile(cb_mask, 0, dst_mask);

                mask_tile_init();
                mask_tile(dst0, dst_mask);
            }

            pack_tile(dst0, cb_exps);
            cb_push_back(cb_exps, onetile);
            REL();
        }

        // step 2, compute 1/sum(exp(x))
        ACQ();
        cb_reserve_back(cb_recipsumexps, onetile);
        reduce_init_delta_v2<false>(REDUCE_OP, REDUCE_DIM);
        for(uint32_t h = 0; h < Ht; ++h) {
            cb_wait_front(cb_exps, h + 1); // must be a cumulative wait for correctness

            constexpr uint32_t bcast_scaler0 = 0; // 0th index from bcast_scaler CB
            reduce_tile_v2(REDUCE_OP, REDUCE_DIM, cb_exps, cb_bcast_scaler, h, bcast_scaler0, dst0);
        }
        reduce_revert_delta_v2();

        recip_tile_init();
        recip_tile(dst0); // DST[0] = 1/sum(exp(x))
        pack_tile(dst0, cb_recipsumexps);
        cb_push_back(cb_recipsumexps, onetile);
        REL();

        // step 3, compute final result
        cb_wait_front(cb_recipsumexps, onetile); // will reuse Wt times for bcast
        // now cb_sumexps has exp tiles, need to multiply by our DST[2]
        // by now we already did a umulative wait for Wt tiles in cb_exps
        mul_bcast_rows_init_short();
        for (uint32_t h = 0; h < Ht; h += onetile) {
            ACQ();
            cb_reserve_back(cb_out0, onetile);

            mul_tiles_bcast<BroadcastType::ROW>(cb_exps, cb_recipsumexps, h, 0, dst0); // tile *= 1/(sum(exp(x)))
            pack_tile(dst0, cb_out0);

            cb_push_back(cb_out0, onetile);
            REL();
        }

        cb_pop_front(cb_recipsumexps, onetile);
        cb_pop_front(cb_exps, Ht);
    }
    cb_pop_front(cb_bcast_scaler, onetile); // comes from the reader
    cb_pop_front(cb_mask, onetile); // comes from the reader
}
}
