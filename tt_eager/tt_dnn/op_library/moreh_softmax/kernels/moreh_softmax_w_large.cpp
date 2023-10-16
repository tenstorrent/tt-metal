// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/softmax.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug_print.h"
#include "tt_eager/tt_dnn/op_library/moreh_softmax/kernels/common_ckernels.hpp"

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
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // step 1
        for (uint32_t w = 0; w < Wt; ++w) {
            // compute exp(x)
            ACQ();
            if (w == Wt - 1) {
                exp_tile_and_mask_tile_to_cb(
                    cb_in0,
                    cb_mask,
                    cb_exps,
                    /*itile=*/0,
                    /*mtile=*/0,
                    /*pop=*/1,
                    /*popm=*/0);
            } else {
                exp_tile_to_cb(cb_in0, cb_exps);
            }
            REL();

            if (w == 0) {
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
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // compute exp(x)
            ACQ();
            exp_tile_to_cb(cb_in0, cb_exps);
            REL();

            // compute exp(x)/sum(exp(x))
            ACQ();
            mul_tiles_bcast_cols_to_cb(cb_exps, cb_recipsumexps, cb_out0, 0, 0, /*pop0=*/1, /*pop1=*/0);
            REL();
        }

        cb_pop_front(cb_recipsumexps, onetile);
    }
}
}  // namespace NAMESPACE
