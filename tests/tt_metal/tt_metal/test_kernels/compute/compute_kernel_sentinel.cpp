// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "api/compute/transpose.h"
#include "api/debug/assert.h"

void kernel_main() {
    SET_CALLED_RECONFIG(RECONFIG_NOTHING_CHANGED);

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // Bfp8_b
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // Bfp16_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;    // Bfp16_b
    constexpr auto cb_out0 = tt::CBIndex::c_16;  // Fp32
    constexpr auto cb_out1 = tt::CBIndex::c_17;  // Bfp8_b

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    // add_init tracks SrcA/SrcB only; record the pack CB the way the matmul checks below do. No
    // mid-kernel compute_kernel_hw_startup here: it re-seeds the sentinel state and would hide the diffs.
    add_init(cb_in0, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_NOTHING_CHANGED));
    add_init(cb_in1, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));
    add_init(cb_in1, cb_in0);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCB));
    add_init(cb_in1, cb_in0);
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));
    add_init(cb_in0, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    matmul_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    matmul_block_init(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    matmul_block_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    bcast_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(cb_in2, cb_in1);
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));

    add_bcast_rows_init(cb_in1, cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_rows_init(cb_in2, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_cols_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    add_bcast_scalar_init(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_bcast_scalar_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_bcast_cols_init(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    mul_bcast_rows_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    sub_bcast_cols_init(cb_in1, cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    sub_bcast_scalar_init(cb_in0, cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));
    binary_tiles_init<false, EltwiseBinaryType::ELWADD>(cb_in2, cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB));

    pack_untilize_dest_init<1>(cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));

    pack_untilize_init(cb_in0, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    reconfig_data_format(cb_in0, cb_in1);
    // REDUCE_ROW+SUM swaps operands: state_configure(icb_scaler=cb_in0, icb=cb_in1, cb_out0)
    // SrcA stays cb_in0 (unchanged from pack_untilize_init above), SrcB and Pack change.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in1, cb_in0, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK));
    reduce_uninit();

    tilize_init(cb_in0, 1, cb_out1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_PACK));

    fast_tilize_init(cb_in2, 1, cb_out0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));
    // fast_tilize_init reprograms the SrcB tile descriptor's num_faces field; balance it with its uninit so
    // that field is restored before a later A/B op (tilizeA_B_reduce_init below) validates the unpacker state.
    fast_tilize_uninit(cb_in2, cb_out0, 1);

    transpose_init(cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    // Migrated off unary_op_init_common (now a deprecated compute_kernel_hw_startup + copy_init
    // forwarder, which is unsafe mid-kernel). The equivalent mid-kernel reconfig sequence produces
    // the same SrcA/Pack diff the sentinel asserts: SrcA only here (pack already cb_out0).
    reconfig_data_format_srca(cb_in0);
    copy_init(cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    // SrcA (cb_in0 -> cb_in1) and Pack (cb_out0 -> cb_out1) both change.
    reconfig_data_format_srca(cb_in1);
    pack_reconfig_data_format(cb_out1);
    // pack_reconfig_data_format is PACK-only and does not reach the sentinel state; record the pack CB
    // the way the matmul checks above do so m_pack_cb tracks cb_out1 and the assert below sees the PACK change.
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    copy_init(cb_in1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK));

    transpose_init(cb_in0);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    copy_init(cb_in2);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));

    tilizeA_B_reduce_init<false, true>(cb_in0, cb_in1, 1);
    ASSERT(TEST_RECONFIG_CALLS(RECONFIG_CHANGED_SRCA));
}
