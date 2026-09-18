// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for BenchmarkCaseFive (worst-case-two init benchmark).
//
// Each of the four single-thread DM reader kernels (one per 3-DFB group) runs
// this kernel, bound to dfb::out0, dfb::out1, dfb::out2.
//
// DFB config: 1Sx4A, num_entries=16, num_producers=1, num_consumers=4
//
// Drains the full 16-entry ring per DFB: the sole DM producer issues all
// 16 implicit reads per DFB.

#include "dfb_implicit_read_helper.h"

namespace {
constexpr uint32_t kNumEntries = 16u;
constexpr uint32_t kReadsPerDm = kNumEntries;
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
