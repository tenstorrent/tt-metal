// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute consumer kernel for BenchmarkCaseSix (worst-case-four init benchmark).
//
// Four Neo threads (Neo0–Neo3) run this kernel. Each Neo checks its NEO_ID CSR
// and drains the DFBs it consumes. DFB IDs use the low-level uint16_t constructor
// (no kernel_bindings_generated.h).
//
// Modeled on eltwise_copy_2_0.cpp / dfb_bench_worst2_compute.cpp:
//   tile_regs_acquire/wait → wait_front → copy_tile → pop_front → commit/release
//
// Host uses entry_size=2048 with default 32×32 tile (no .tile override).
//
// Drains the full ring per DFB (num_entries=1): 1 tile per DFB.
// Producer finish() is called by the DM kernel after each implicit read.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint32_t kTilesPerDfb = 1u;
}  // namespace

#define DRAIN_ALL_DFB(dfb_id)                                          \
    do {                                                               \
        DataflowBuffer _dfb(dfb_id);                                   \
    } while (0)

void kernel_main() {
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();

    unary_op_init_common(0, 0);

    if (neo_id == 0) {
        DRAIN_ALL_DFB(0);
        DRAIN_ALL_DFB(1);
        DRAIN_ALL_DFB(2);
        DRAIN_ALL_DFB(3);
        DRAIN_ALL_DFB(4);
        DRAIN_ALL_DFB(5);
        DRAIN_ALL_DFB(6);
        DRAIN_ALL_DFB(7);
        DRAIN_ALL_DFB(22);
        DRAIN_ALL_DFB(23);
    } else if (neo_id == 1) {
        DRAIN_ALL_DFB(8);
        DRAIN_ALL_DFB(9);
        DRAIN_ALL_DFB(10);
        DRAIN_ALL_DFB(11);
        DRAIN_ALL_DFB(12);
        DRAIN_ALL_DFB(13);
        DRAIN_ALL_DFB(14);
        DRAIN_ALL_DFB(15);
        DRAIN_ALL_DFB(16);
        DRAIN_ALL_DFB(17);
        DRAIN_ALL_DFB(18);
        DRAIN_ALL_DFB(19);
    } else if (neo_id == 2) {
        DRAIN_ALL_DFB(0);
        DRAIN_ALL_DFB(1);
        DRAIN_ALL_DFB(2);
        DRAIN_ALL_DFB(3);
        DRAIN_ALL_DFB(4);
        DRAIN_ALL_DFB(5);
        DRAIN_ALL_DFB(6);
        DRAIN_ALL_DFB(7);
    } else {
        DRAIN_ALL_DFB(8);
        DRAIN_ALL_DFB(9);
        DRAIN_ALL_DFB(10);
        DRAIN_ALL_DFB(11);
        DRAIN_ALL_DFB(12);
        DRAIN_ALL_DFB(13);
        DRAIN_ALL_DFB(14);
        DRAIN_ALL_DFB(15);
        DRAIN_ALL_DFB(20);
        DRAIN_ALL_DFB(21);
    }
}
