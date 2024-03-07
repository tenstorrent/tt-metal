// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_mask = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_add = tt::CB::c_intermed2;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;

    binary_op_init_common(cb_in0, cb_bcast_scaler);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // step 1, compute exp(x)
        for (uint32_t h = 0; h < Ht; ++h) {
            ACQ();
            if (h == Ht - 1) {
                #ifdef SOFTMAX
                    exp_tile_and_mask_tile_to_cb(
                        cb_in0,
                        cb_mask,
                        cb_exps,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);
                #else
                    rexp_tile_and_mask_tile_to_cb(
                        cb_in0,
                        cb_mask,
                        cb_exps,
                        /*itile=*/0,
                        /*mtile=*/0,
                        /*pop=*/1,
                        /*popm=*/0);
                #endif
            } else {
                #ifdef SOFTMAX
                    exp_tile_to_cb(cb_in0, cb_exps);
                #else
                    rexp_tile_to_cb(cb_in0, cb_exps);
                #endif
            }
            REL();

            if (h == 0) {
                ACQ();
                copy_tile_to_cb(cb_exps, cb_add);
                REL();
            } else {
                ACQ();
                add_tiles_to_cb(cb_add, cb_exps, cb_add);
                REL();
            }
        }

        // step 2, compute 1/sum(exp(x))
        ACQ();
        reduce_tile_and_recip_tile_to_cb(
            REDUCE_OP, REDUCE_DIM, cb_add, cb_bcast_scaler, cb_recipsumexps, /*size=*/1, /*pop0=*/1, /*pop1=*/0);
        REL();

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x)
            ACQ();
            #ifdef SOFTMAX
                exp_tile_to_cb(cb_in0, cb_exps);
            #else
                rexp_tile_to_cb(cb_in0, cb_exps);
            #endif
            REL();

            // compute exp(x)/sum(exp(x))
            ACQ();
            #ifdef LOG
                mul_tiles_bcast_rows_log_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
            #else
                mul_tiles_bcast_rows_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
            #endif
            REL();
        }
        cb_pop_front(cb_recipsumexps, onetile);
    }
}
}  // namespace NAMESPACE
