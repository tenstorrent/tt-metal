// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark reader DM kernel for the worst-case DFB ISR latency benchmark.
//
// Acts as PRODUCER on three DM→Tensix DFBs (4Sx4A each):
//   dfb::out0  - 4Sx4A DM→Tensix
//   dfb::out1  - 4Sx4A DM→Tensix
//   dfb::out2  - 4Sx4A DM→Tensix
//
// Three concurrent 4Sx4A DFBs represent the hardware worst case:
//   - TCs:      3 DFBs × 5 TCs/tensix = 15/16 DM-visible TCs per tensix used
//               (floor(16/5) = 3 is the maximum; a 4th DFB would overflow tensix 0-3)
//   - Remapper: 3 DFBs × 4 producers × 1-to-4 fan-out = 12/16 1-to-many entries,
//               48 set_clientR_slot writes (maximum achievable under the TC constraint)
//
// Threshold table (each 4Sx4A DFB, num_entries=16, num_producers=4):
//   num_txn_ids   = 2  (smallest n≥2 s.t. 16 % (n × 4 × 1) == 0)
//   hw_threshold  = 8  (16 / 2 = 8 total reads per txn ID across all 4 DMs)
//   per_txn       = 2  (hw_threshold / num_producers = 8 / 4)
//   tiles_to_post = 2  (per_txn / num_tcs_per_risc = 2 / 1)
//
// Each DM issues per_txn=2 reads per DFB. All 4 DMs together contribute 8
// total reads → ISR fires → posts 2 credits to each of the 4 Neo consumer TCs.
//
// Source: DRAM bank 0 offset 0 (data content irrelevant; ISR timing is the goal).

#include "dfb_implicit_read_helper.h"

void kernel_main() {
    Noc noc;
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer out1(dfb::out1);
    DataflowBuffer out2(dfb::out2);

    // per_txn = 2: each DM issues 2 reads per DFB (4 DMs × 2 = 8 = hw_threshold)
    // dfb_issue_implicit_read(noc, out0);
    // dfb_issue_implicit_read(noc, out0);

    // dfb_issue_implicit_read(noc, out1);
    // dfb_issue_implicit_read(noc, out1);

    // dfb_issue_implicit_read(noc, out2);
    // dfb_issue_implicit_read(noc, out2);

    // out0.finish();
    // out1.finish();
    // out2.finish();
}
