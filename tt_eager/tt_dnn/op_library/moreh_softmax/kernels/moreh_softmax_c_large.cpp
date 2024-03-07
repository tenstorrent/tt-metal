// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_recipsumexps = tt::CB::c_intermed1;
    constexpr auto cb_add = tt::CB::c_intermed2;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps);

    for (uint32_t n = 0; n < N; ++n) {
        // step 1, compute exp(x)
        for (uint32_t i = 0; i < dim_size; ++i) {
            ACQ();
            #ifdef SOFTMAX
                exp_tile_to_cb(cb_in0, cb_exps);
            #else
                rexp_tile_to_cb(cb_in0, cb_exps);
            #endif
            REL();

            if (i == 0) {
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
        recip_tile_to_cb(cb_add, cb_recipsumexps);
        REL();

        // step 3, compute final result
        cb_wait_front(cb_recipsumexps, onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
            // compute exp(x)
            ACQ();
            #ifdef SOFTMAX
                exp_tile_to_cb(cb_in0, cb_exps);
            #else
                rexp_tile_to_cb(cb_in0, cb_exps);
            #endif
            REL();

            // multiply recip
            ACQ();
            #ifdef LOG
                mul_tiles_log_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
            #else
                mul_tiles_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
            #endif
            REL();
        }

        cb_pop_front(cb_recipsumexps, onetile);
    }
}
}  // namespace NAMESPACE
