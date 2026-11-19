// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t kMaxChecks = 64;

void kernel_main() {
    // Stop after this many checks (bisects a hang); all checks run when it is >= the number of checks.
    constexpr uint32_t max_checks = get_arg(args::max_checks);
    SET_CALLED_RECONFIG(RECONFIG_NOTHING_CHANGED);

    // Each check records (expected << 8) | recorded bits; the PACK thread writes them to `status` for the host.
    uint32_t results[kMaxChecks] = {};
    uint32_t num_checks = 0;
#define CHECK(expected)                                                                             \
    do {                                                                                            \
        const uint32_t recorded = ComputeKernelSentinel::instance().recorded_calls();               \
        (void)TEST_RECONFIG_CALLS(expected);                                                        \
        if (num_checks < kMaxChecks) {                                                              \
            results[num_checks] = (static_cast<uint32_t>(expected) << 8) | recorded;                \
        }                                                                                           \
        ++num_checks;                                                                               \
        if (num_checks >= max_checks) {                                                             \
            report();                                                                               \
            return;                                                                                 \
        }                                                                                           \
    } while (0)

    DataflowBuffer dfb_in0(dfb::in0);    // MxFp8P
    DataflowBuffer dfb_in1(dfb::in1);    // Float16_b
    DataflowBuffer dfb_in2(dfb::in2);    // Float16_b
    DataflowBuffer dfb_out0(dfb::out0);  // Float16_b
    DataflowBuffer dfb_out1(dfb::out1);  // MxFp8P
    DataflowBuffer dfb_status(dfb::status);
    const uint32_t cb_in0 = dfb_in0.get_id();
    const uint32_t cb_in1 = dfb_in1.get_id();
    const uint32_t cb_in2 = dfb_in2.get_id();
    const uint32_t cb_out0 = dfb_out0.get_id();
    const uint32_t cb_out1 = dfb_out1.get_id();
    auto report = [&]() {
        // Report: word 0 = number of checks, then one word per check.
        // The checks leave the packer configured for whatever CB the sentinel tracked last; switch to status.
        pack_reconfig_data_format(dfb_status.get_id());
        pack_init(dfb_status.get_id());
        dfb_status.reserve_back(1);
        dummy_pack(dfb_status.get_id());  // TEN-4746: a PACR after reserve_back orders the push_back below.
#ifdef UCK_CHLKC_PACK
        ckernel::tensix_sync();
        volatile uint32_t* status = reinterpret_cast<volatile uint32_t*>(dfb_status.get_write_ptr() << 4);
        status[0] = num_checks;
        for (uint32_t i = 0; i < kMaxChecks; ++i) {
            status[1 + i] = results[i];
        }
#endif
        dfb_status.push_back(1);
    };

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    // add_init tracks SrcA/SrcB only; record the pack CB the way the matmul checks below do. No
    // mid-kernel compute_kernel_hw_startup here: it re-seeds the sentinel state and would hide the diffs.
    add_init(cb_in0, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    CHECK(RECONFIG_NOTHING_CHANGED);
    add_init(cb_in1, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    CHECK(RECONFIG_CHANGED_SRCA);
    add_init(cb_in1, cb_in0);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    CHECK(RECONFIG_CHANGED_SRCB);
    add_init(cb_in1, cb_in0);
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    CHECK(RECONFIG_CHANGED_PACK);
    add_init(cb_in0, cb_in1);
    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);

    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in2);
    CHECK(RECONFIG_CHANGED_SRCA);

    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    matmul_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);

    matmul_block_init(cb_in1, cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);

    state_configure<Operand::PACK>(cb_out0, __builtin_LINE());
    matmul_block_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);

    bcast_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(cb_in2, cb_in1);
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);

    add_bcast_rows_init(cb_in1, cb_in2);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    add_bcast_rows_init(cb_in2, cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    add_bcast_cols_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    add_bcast_scalar_init(cb_in1, cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    mul_bcast_scalar_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    mul_bcast_cols_init(cb_in1, cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    mul_bcast_rows_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    sub_bcast_cols_init(cb_in1, cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    sub_bcast_scalar_init(cb_in0, cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);
    binary_tiles_init<false, EltwiseBinaryType::ELWADD>(cb_in2, cb_in2);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_SRCB);

    pack_untilize_dest_init<1>(cb_out0);
    CHECK(RECONFIG_CHANGED_PACK);

    pack_untilize_init(cb_in0, cb_out1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK);

    reconfig_data_format(cb_in0, cb_in1);
    // REDUCE_ROW+SUM swaps operands: state_configure(icb_scaler=cb_in0, icb=cb_in1, cb_out0)
    // SrcA stays cb_in0 (unchanged from pack_untilize_init above), SrcB and Pack change.
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in1, cb_in0, cb_out0);
    CHECK(RECONFIG_CHANGED_SRCB | RECONFIG_CHANGED_PACK);
    reduce_uninit();

    tilize_init(cb_in0, 1, cb_out1);
    CHECK(RECONFIG_CHANGED_PACK);

    fast_tilize_init(cb_in2, 1, cb_out0);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK);
    // fast_tilize_init reprograms the SrcB tile descriptor's num_faces field; balance it with its uninit so
    // that field is restored before a later A/B op (tilizeA_B_reduce_init below) validates the unpacker state.
    fast_tilize_uninit(cb_in2, cb_out0, 1);

    transpose_init(cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA);

    // Migrated off unary_op_init_common (a now-removed compute_kernel_hw_startup + copy_init
    // forwarder, which was unsafe mid-kernel). The equivalent mid-kernel reconfig sequence produces
    // the same SrcA/Pack diff the sentinel asserts: SrcA only here (pack already cb_out0).
    reconfig_data_format_srca(cb_in0);
    copy_init(cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA);

    // SrcA (cb_in0 -> cb_in1) and Pack (cb_out0 -> cb_out1) both change.
    reconfig_data_format_srca(cb_in1);
    pack_reconfig_data_format(cb_out1);
    // pack_reconfig_data_format is PACK-only and does not reach the sentinel state; record the pack CB
    // the way the matmul checks above do so m_pack_cb tracks cb_out1 and the assert below sees the PACK change.
    state_configure<Operand::PACK>(cb_out1, __builtin_LINE());
    copy_init(cb_in1);
    CHECK(RECONFIG_CHANGED_SRCA | RECONFIG_CHANGED_PACK);

    transpose_init(cb_in0);
    CHECK(RECONFIG_CHANGED_SRCA);

    copy_init(cb_in2);
    CHECK(RECONFIG_CHANGED_SRCA);

    tilizeA_B_reduce_init<false, true>(cb_in0, cb_in1, 1);
    CHECK(RECONFIG_CHANGED_SRCA);
    report();
}
