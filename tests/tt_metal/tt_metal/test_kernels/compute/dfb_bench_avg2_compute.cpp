// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for BenchmarkCaseThree (average-case-two init benchmark).
//
// Modeled on eltwise_copy_2_0.cpp (reader_datacopy_writer in test_direct.cpp):
//   tile_regs_acquire/wait → wait_front → copy_tile → pop_front → commit/release
//
// Acts as STRIDED CONSUMER on three 4Sx4S DM→Tensix DFBs:
//   dfb::in0, dfb::in1, dfb::in2
//
// Drains the full 16-entry ring per DFB (num_entries=16, 4 Neos):
//   4 tiles per Neo per DFB (1 strided TC each)
//
// finish() is called by the reader DM kernels (producer side).

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kNumNeos = 4u;
constexpr uint32_t kTilesPerNeo = kNumEntries / kNumNeos;
}  // namespace

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer in2(dfb::in2);

    unary_op_init_common(dfb::in0, dfb::in0);
    for (uint32_t i = 0; i < kTilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        in0.wait_front(1);
        copy_tile(dfb::in0, 0, 0);
        in0.pop_front(1);

        tile_regs_commit();
        tile_regs_release();
    }

    unary_op_init_common(dfb::in1, dfb::in1);
    for (uint32_t i = 0; i < kTilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        in1.wait_front(1);
        copy_tile(dfb::in1, 0, 0);
        in1.pop_front(1);

        tile_regs_commit();
        tile_regs_release();
    }

    unary_op_init_common(dfb::in2, dfb::in2);
    for (uint32_t i = 0; i < kTilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        in2.wait_front(1);
        copy_tile(dfb::in2, 0, 0);
        in2.pop_front(1);

        tile_regs_commit();
        tile_regs_release();
    }
}
