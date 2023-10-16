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
#include "tt_eager/tt_dnn/op_library/moreh_softmax_backward/kernels/common_ckernels.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_ydy = tt::CB::c_intermed0;  // y * dy
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_dy_m_sum = tt::CB::c_intermed2;  // dy - sum

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y);

    constexpr int dst0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            ACQ();
            mul_tiles_to_cb(cb_y, cb_dy, cb_ydy);
            REL();

            if (i == 0) {
                ACQ();
                copy_tile_to_cb(cb_ydy, cb_sum);
                REL();
            } else {
                ACQ();
                add_tiles_to_cb(cb_sum, cb_ydy, cb_sum);
                REL();
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            ACQ();
            sub_tiles_to_cb(
                cb_dy,
                cb_sum,
                cb_dy_m_sum,
                /*itile0=*/0,
                /*itile1=*/0,
                /*pop0=*/1,
                /*pop1=*/0);
            REL();

            // (dy - sum) * y
            ACQ();
            mul_tiles_to_cb(cb_dy_m_sum, cb_y, cb_dx);
            REL();
        }
        cb_pop_front(cb_sum, onetile);
    }
}
}  // namespace NAMESPACE
