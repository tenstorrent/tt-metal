// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for the average-case DFB ISR latency benchmark.
//
// Modeled on eltwise_copy_2_0.cpp (reader_datacopy_writer in test_direct.cpp):
//   tile_regs_acquire/wait → wait_front → copy_tile → pop_front → commit/release
//
// Participates in three DFBs:
//   dfb::ss_in   - DM→Tensix, 4Sx4S STRIDED consumer (paired with reader_dm ss_out)
//   dfb::sa_in   - DM→Tensix, 4Sx4A ALL consumer     (paired with reader_dm sa_out)
//   dfb::t6_out  - Tensix→DM, STRIDED producer        (paired with writer_dm t6_in)
//
// Drains the full 16-entry ring (num_entries=16, 4 Neos):
//   ss_in:  4 tiles per Neo (1 strided TC)
//   sa_in: 16 tiles per Neo (4 ALL TCs × 4 entries each)
//   t6_out: 4 tiles per Neo (reserve/push, no upstream copy)
//
// finish() is called by the reader (ss/sa producer) and writer (t6 consumer) kernels.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kNumNeos = 4u;
constexpr uint32_t kSaTcsPerNeo = 4u;
constexpr uint32_t kTilesPerNeo = kNumEntries / kNumNeos;
constexpr uint32_t kSaTilesPerNeo = kTilesPerNeo * kSaTcsPerNeo;
constexpr uint32_t kT6TilesPerNeo = kTilesPerNeo;
}  // namespace

void kernel_main() {
    DataflowBuffer ss_in(dfb::ss_in);
    DataflowBuffer sa_in(dfb::sa_in);
    DataflowBuffer t6_out(dfb::t6_out);

    unary_op_init_common(dfb::ss_in, dfb::ss_in);
    for (uint32_t i = 0; i < kTilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        ss_in.wait_front(1);
        copy_tile(dfb::ss_in, 0, 0);
        ss_in.pop_front(1);

        tile_regs_commit();
        tile_regs_release();
    }

    unary_op_init_common(dfb::sa_in, dfb::sa_in);
    for (uint32_t i = 0; i < kSaTilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        sa_in.wait_front(1);
        copy_tile(dfb::sa_in, 0, 0);
        sa_in.pop_front(1);

        tile_regs_commit();
        tile_regs_release();
    }

    unary_op_init_common(dfb::t6_out, dfb::t6_out);
    for (uint32_t i = 0; i < kT6TilesPerNeo; i++) {
        tile_regs_acquire();
        tile_regs_wait();

        t6_out.reserve_back(1);
        t6_out.push_back(1);

        tile_regs_commit();
        tile_regs_release();
    }
}
