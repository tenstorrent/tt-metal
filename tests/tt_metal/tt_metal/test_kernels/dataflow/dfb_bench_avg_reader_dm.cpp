// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for the average-case DFB ISR latency benchmark.
//
// Acts as PRODUCER on two DFBs:
//   dfb::ss_out  - DM→Tensix, 4Sx4S (strided producer x strided consumer)
//   dfb::sa_out  - DM→Tensix, 4Sx4A (strided producer x ALL consumer)
//
// Threshold table (both DFBs, num_entries=16, num_producers=4):
//   num_txn_ids   = 2  (smallest n≥2 s.t. 16 % (n × 4 × 1) == 0)
//   hw_threshold  = 8  (16 / 2 = 8 total reads per txn ID across all 4 DMs)
//   per_txn       = 2  (hw_threshold / num_producers = 8 / 4)
//   tiles_to_post = 2  (per_txn / num_tcs_per_risc = 2 / 1)
//
// Each DM issues per_txn=2 reads per DFB. All 4 DMs together contribute 8
// total reads → ISR fires → posts 2 credits to each Tensix consumer TC.
//
// Source: DRAM bank 0 offset 0 (data content irrelevant; ISR timing is the goal).

#include "dfb_implicit_read_helper.h"

void kernel_main() {
    Noc noc;
    DataflowBuffer dfb_ss(dfb::ss_out);
    DataflowBuffer dfb_sa(dfb::sa_out);

    // per_txn = 2: each DM issues 2 reads per DFB (4 DMs × 2 = 8 = hw_threshold)
    dfb_issue_implicit_read(noc, dfb_ss);
    dfb_issue_implicit_read(noc, dfb_ss);

    dfb_issue_implicit_read(noc, dfb_sa);
    dfb_issue_implicit_read(noc, dfb_sa);

    // dfb_ss.finish();
    // dfb_sa.finish();
}
