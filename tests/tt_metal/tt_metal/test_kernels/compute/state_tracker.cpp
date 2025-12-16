// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/untilize.h"
#include "debug/assert.h"  // Required in all kernels using watcher asserts
#include "compute_kernel_api/state_tracker.h"
#include "compute_kernel_api/reduce.h"

namespace NAMESPACE {
void MAIN {
    SET_CALLED_RECONFIG(RECONFIG_NOTHING_CHANGED);

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Bfp8_b
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Bfp16_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // Bfp16_b
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Fp32
    constexpr auto cb_out1 = tt::CBIndex::c_17;  // Bfp8_b

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    // Matmul init short: tests state tracker with 2-param version (no PACK)
    mm_init_short(cb_in2, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    mm_block_init_short(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    init_bcast<ELWADD, BroadcastType::NONE>(cb_in2, cb_in0, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    add_bcast_rows_init_short(cb_in1, cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_rows_init_short(cb_in2, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_cols_init_short(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_scalar_init_short(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_tiles_bcast_scalar_init_short(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_bcast_cols_init_short(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_bcast_rows_init_short(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    sub_bcast_cols_init_short(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    sub_tiles_bcast_scalar_init_short(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    binary_tiles_init<false, ELWADD>(cb_in2, cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    pack_untilize_dest_init<1>(cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));

    pack_untilize_init(cb_in0, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    reduce_init(cb_in1, cb_in0, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    tilize_init(cb_in0, 1, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    tilize_init_no_pack(cb_in1, 1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    fast_tilize_init(cb_in2, 1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    transpose_wh_init_short(cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    untilize_init(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    // All commented code: Breaks functionality, will be addressed in issue #34432
    // unary_op_init_common(cb_in0, cb_out0);
    // unary_op_init_common(cb_in1, cb_out1);
    // unary_op_init_common_no_pack(cb_in2);
    // unary_bcast_init
    // binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // mm_init(cb_in2, cb_in0, cb_out0);
    // mm_block_init(cb_in0, cb_in1, cb_out0);
    // #if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)
    //     tilizeA_B_reduce_init<false, true>(cb_in0, cb_in1, 1, cb_out0);
    // #endif
    // binary_dest_reuse_tiles_init(cb_in1);
    // transpose_wh_init(cb_in0, cb_out0);
}
}  // namespace NAMESPACE
