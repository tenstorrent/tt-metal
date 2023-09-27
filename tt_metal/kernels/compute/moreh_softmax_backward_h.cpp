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

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_mask = tt::CB::c_in3;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_ydy = tt::CB::c_intermed0; // y * dy
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_inter2 = tt::CB::c_intermed2;

    binary_op_init_common(cb_y, cb_bcast_scaler);

    cb_wait_front(cb_bcast_scaler, onetile); // comes from the reader
    cb_wait_front(cb_mask, onetile); // comes from the reader

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    constexpr int dst0 = 0;
    for(uint32_t n = 0; n < N; ++n) {
        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h)
        {
            ACQ();
            cb_reserve_back(cb_ydy, onetile);

            cb_wait_front(cb_y, h + 1);
            cb_wait_front(cb_dy, h + 1);

            mul_tiles_init();
            mul_tiles(cb_y, cb_dy, h, h, dst0);

            if (h == Ht - 1) {
                constexpr int dst_mask = 1;
                copy_tile_init();
                copy_tile(cb_mask, 0, dst_mask);

                mask_tile_init();
                mask_tile(dst0, dst_mask);
            }
            pack_tile(dst0, cb_ydy);
            cb_push_back(cb_ydy, onetile);

            REL();
        }

        // step 2, compute sum(y * dy)
        ACQ();
        cb_reserve_back(cb_sum, onetile);
        reduce_init_delta_v2<false>(REDUCE_OP, REDUCE_DIM);
        for (uint32_t h = 0; h < Ht; ++h)
        {
            cb_wait_front(cb_ydy, h + 1); // must be a cumulative wait for correctness

            constexpr uint32_t bcast_scaler0 = 0; // 0th index from bcast_scaler CB
            reduce_tile_v2(REDUCE_OP, REDUCE_DIM, cb_ydy, cb_bcast_scaler, h, bcast_scaler0, dst0);
        }
        cb_pop_front(cb_ydy, Ht);

        reduce_revert_delta_v2();
        pack_tile(dst0, cb_sum);
        cb_push_back(cb_sum, onetile);
        REL();

        // step 3, compute final result
        cb_wait_front(cb_sum, onetile); // will reuse Wt times for bcast
        for (uint32_t h = 0; h < Ht; ++h)
        {
            // dy - sum
            ACQ();

            cb_reserve_back(cb_inter2, onetile);
            cb_wait_front(cb_dy, h + 1);

            // sub_bcast_rows_init_short();
            {
                MATH(( llk_math_eltwise_binary_init<ELWSUB, BroadcastType::ROW, MATH_FIDELITY>() ));
                UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>() ));
            }
            sub_tiles_bcast<BroadcastType::ROW>(cb_dy, cb_sum, h, 0, dst0);

            pack_tile(dst0, cb_inter2);
            cb_push_back(cb_inter2, onetile);

            REL();

            // (dy - sum) * y
            ACQ();

            cb_reserve_back(cb_dx, onetile);
            cb_wait_front(cb_y, h + 1);
            cb_wait_front(cb_inter2, onetile);

            mul_tiles_init();
            mul_tiles(cb_y, cb_inter2, h, 0, dst0);

            pack_tile(dst0, cb_dx);

            cb_pop_front(cb_inter2, onetile);
            cb_push_back(cb_dx, onetile);

            REL();
        }

        cb_pop_front(cb_sum, onetile);
        cb_pop_front(cb_dy, Ht);
        cb_pop_front(cb_y, Ht);
    }
}
}
