// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/untilize.h"
#include "api/debug/assert.h"

void kernel_main() {
    SET_CALLED_RECONFIG(RECONFIG_NOTHING_CHANGED);

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Bfp8_b
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Bfp16_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // Bfp16_b
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Fp32
    constexpr auto cb_out1 = tt::CBIndex::c_17;  // Bfp8_b

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_NOTHING_CHANGED));
    binary_op_init_common(cb_in1, cb_in1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));
    binary_op_init_common(cb_in1, cb_in0, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCB));
    binary_op_init_common(cb_in1, cb_in0, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    binary_dest_reuse_tiles_init(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    mm_init(cb_in0, cb_in1, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    mm_block_init_short(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    mm_block_init(cb_in0, cb_in1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    init_bcast<ELWADD, BroadcastType::NONE>(cb_in2, cb_in1, cb_out1);
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

    fast_tilize_init(cb_in2, 1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    transpose_wh_init_short(cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    untilize_init(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    unary_op_init_common(cb_in0, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    unary_op_init_common(cb_in1, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    transpose_wh_init(cb_in0, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    copy_tile_to_dst_init_short(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    tilizeA_B_reduce_init<false, true>(cb_in0, cb_in1, 1, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));
}
