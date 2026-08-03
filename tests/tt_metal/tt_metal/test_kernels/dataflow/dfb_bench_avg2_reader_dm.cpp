// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for BenchmarkCaseThree (average-case-two init benchmark).
//
// Acts as STRIDED PRODUCER on three 4Sx4S DM→Tensix DFBs:
//   dfb::out0, dfb::out1, dfb::out2
//
// Drains the full 16-entry ring per DFB (num_entries=16, 4 strided producers):
//   each DM issues 16 / 4 = 4 implicit reads per DFB.

#include "dfb_implicit_read_helper.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kNumProducers = 4u;
constexpr uint32_t kReadsPerDm = kNumEntries / kNumProducers;
}  // namespace

void kernel_main() {
    Noc noc;
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer out1(dfb::out1);
    DataflowBuffer out2(dfb::out2);

    for (uint32_t i = 0; i < kReadsPerDm; i++) {
        dfb_issue_implicit_read(noc, out0);
        dfb_issue_implicit_read(noc, out1);
        dfb_issue_implicit_read(noc, out2);
    }

    out0.finish();
    out1.finish();
    out2.finish();
}
