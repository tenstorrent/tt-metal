// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for the worst-case-two DFB ISR latency benchmark.
//
// Each of the four single-thread DM reader kernels (one per 3-DFB group) runs
// this kernel, bound to dfb::out0, dfb::out1, dfb::out2.
//
// DFB config: 1Sx4A, num_entries=16, num_producers=1, num_consumers=4
//
// Threshold table:
//   num_txn_ids   = 2  (smallest n≥2 s.t. 16 % (n × 1 × 1) == 0)
//   hw_threshold  = 8  (16 / 2 = 8 total reads per txn ID)
//   per_txn       = 8  (hw_threshold / num_producers = 8 / 1, single DM)
//   tiles_to_post = 8  (per_txn / num_tcs_per_risc = 8 / 1)
//
// This kernel issues per_txn=8 reads per DFB to hit the ISR threshold once,
// causing the ISR to fire and post 8 credits to each of the 4 Neo consumer TCs.
// After issuing all reads, finish() drains the DFB.
//
// Source: DRAM bank 0 offset 0 (data content irrelevant; ISR timing is the goal).

#include "dfb_implicit_read_helper.h"

void kernel_main() {
    Noc noc;
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer out1(dfb::out1);
    DataflowBuffer out2(dfb::out2);

    // per_txn = 8: each single DM must issue 8 reads per DFB (hw_threshold = 8)
    for (int i = 0; i < 8; i++) {
        dfb_issue_implicit_read(noc, out0);
        dfb_issue_implicit_read(noc, out1);
        dfb_issue_implicit_read(noc, out2);
    }

    // out0.finish();
    // out1.finish();
    // out2.finish();
}
