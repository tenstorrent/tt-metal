// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for BenchmarkCaseFive (worst-case-two init benchmark).
//
// Modeled on eltwise_copy_2_0.cpp (reader_datacopy_writer in test_direct.cpp):
//   tile_regs_acquire/wait → wait_front → copy_tile → pop_front → commit/release
//
// Single compute kernel with num_threads=4 (Neo0–Neo3). Acts as ALL consumer on
// 12 DFBs (12 × 1Sx4A): every Neo receives every DFB via 1-to-4 remapper fan-out.
//
// Drains the full 16-entry ring per DFB (num_entries=16, 1 producer):
//   16 tiles per Neo per DFB (1 ALL TC × 16 entries)
//
// finish() is called by the reader DM kernels (producer side).

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kTilesPerNeoPerDfb = kNumEntries;
}  // namespace

#define DRAIN_ALL_DFB(dfb_handle, dfb_operand)                       \
    do {                                                             \
        unary_op_init_common(dfb_operand, dfb_operand);              \
        for (uint32_t _i = 0; _i < kTilesPerNeoPerDfb; _i++) {       \
            tile_regs_acquire();                                     \
            tile_regs_wait();                                        \
            (dfb_handle).wait_front(1);                              \
            copy_tile(dfb_operand, 0, 0);                            \
            (dfb_handle).pop_front(1);                               \
            tile_regs_commit();                                      \
            tile_regs_release();                                     \
        }                                                            \
    } while (0)

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer in2(dfb::in2);
    DataflowBuffer in3(dfb::in3);
    DataflowBuffer in4(dfb::in4);
    DataflowBuffer in5(dfb::in5);
    DataflowBuffer in6(dfb::in6);
    DataflowBuffer in7(dfb::in7);
    DataflowBuffer in8(dfb::in8);
    DataflowBuffer in9(dfb::in9);
    DataflowBuffer in10(dfb::in10);
    DataflowBuffer in11(dfb::in11);

    DRAIN_ALL_DFB(in0, dfb::in0);
    DRAIN_ALL_DFB(in1, dfb::in1);
    DRAIN_ALL_DFB(in2, dfb::in2);
    DRAIN_ALL_DFB(in3, dfb::in3);
    DRAIN_ALL_DFB(in4, dfb::in4);
    DRAIN_ALL_DFB(in5, dfb::in5);
    DRAIN_ALL_DFB(in6, dfb::in6);
    DRAIN_ALL_DFB(in7, dfb::in7);
    DRAIN_ALL_DFB(in8, dfb::in8);
    DRAIN_ALL_DFB(in9, dfb::in9);
    DRAIN_ALL_DFB(in10, dfb::in10);
    DRAIN_ALL_DFB(in11, dfb::in11);
}
