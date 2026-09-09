// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for the average-case DFB ISR latency benchmark.
//
// Acts as PRODUCER on two DFBs:
//   dfb::ss_out  - DM→Tensix, 4Sx4S (strided producer x strided consumer)
//   dfb::sa_out  - DM→Tensix, 4Sx4A (strided producer x ALL consumer)
//
// Drains the full 16-entry ring (num_entries=16, 4 strided producers):
//   each DM issues 16 / 4 = 4 implicit reads per DFB.

#include "dfb_implicit_read_helper.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kNumProducers = 4u;
constexpr uint32_t kReadsPerDm = kNumEntries / kNumProducers;
}  // namespace

void kernel_main() {
    Noc noc;
    DataflowBuffer dfb_ss(dfb::ss_out);
    DataflowBuffer dfb_sa(dfb::sa_out);

    for (uint32_t i = 0; i < kReadsPerDm; i++) {
        dfb_issue_implicit_read(noc, dfb_ss);
        dfb_issue_implicit_read(noc, dfb_sa);
    }

    dfb_ss.finish();
    dfb_sa.finish();
}
