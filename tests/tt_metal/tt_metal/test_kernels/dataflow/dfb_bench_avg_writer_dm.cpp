// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark writer DM kernel for the average-case DFB ISR latency benchmark.
//
// Acts as CONSUMER on one DFB:
//   dfb::t6_in  - Tensix→DM, 4Sx4S (strided producer × strided consumer)
//
// Drains the full 16-entry ring (num_entries=16, 2 strided consumers):
//   each writer pops 16 / 2 = 8 tiles.

#include "api/dataflow/dataflow_buffer.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kNumConsumers = 2u;
constexpr uint32_t kTilesPerWriter = kNumEntries / kNumConsumers;
}  // namespace

void kernel_main() {
    DataflowBuffer dfb(dfb::t6_in);

    for (uint32_t i = 0; i < kTilesPerWriter; i++) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }

    dfb.finish();
}
