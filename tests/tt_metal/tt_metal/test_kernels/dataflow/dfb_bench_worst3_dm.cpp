// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM producer kernel for BenchmarkCaseSeven ISR latency benchmark.
//
// Six DM threads (DM0–DM5) run this kernel. Each DM checks its mhartid and
// issues one implicit NOC read per DFB it produces, then calls finish().
// DFB IDs are addressed via the low-level uint16_t constructor since
// kernel_bindings_generated.h has no constants (DFBs are created with
// explicit risc masks, not kernel bindings).
//
// DFB layout (32 DFBs total, 16×1Sx1S + 16×1Sx2A):
//   1Sx1S (DFBs 0–15): one DM producer, one Neo STRIDED consumer
//     DFB  0– 3 : DM4 → Neo0
//     DFB  4– 7 : DM5 → Neo1
//     DFB  8–11 : DM4 → Neo2
//     DFB 12–15 : DM5 → Neo3
//
//   1Sx2A (DFBs 16–31): one DM producer, two Neo ALL consumers via remapper
//     DFB 16–19 : DM2 → {Neo0, Neo2}
//     DFB 20–23 : DM3 → {Neo0, Neo2}
//     DFB 24–27 : DM4 → {Neo1, Neo3}
//     DFB 28–31 : DM5 → {Neo1, Neo3}
//
// TC budget (16 per tensix, 64 total):
//   t0: 4(1Sx1S Neo0) + 4(DM4 prod 24-27) + 4(Neo0 cons DM2) + 4(Neo0 cons DM3) = 16
//   t1: 4(1Sx1S Neo1) + 4(DM5 prod 28-31) + 4(Neo1 cons DM4) + 4(Neo1 cons DM5) = 16
//   t2: 4(1Sx1S Neo2) + 4(DM2 prod 16-19) + 4(Neo2 cons DM2) + 4(Neo2 cons DM3) = 16
//   t3: 4(1Sx1S Neo3) + 4(DM3 prod 20-23) + 4(Neo3 cons DM4) + 4(Neo3 cons DM5) = 16
// Remapper: 16 × 1-to-2 entries (all 16 one-to-many slots), 32 set_clientR_slot writes.
//
// Threshold table (all DFBs, num_entries=1, num_producers=1):
//   num_txn_ids   = 1  (no n≥2 divides 1; falls back to 1)
//   hw_threshold  = 1  (1 / 1 = 1 total read per txn ID)
//   per_txn       = 1  (each DM issues exactly 1 read to fire the ISR)
//   tiles_to_post = 1
//
// Each DM issues 1 read per DFB → ISR fires → posts 1 credit to consumer TC.
//
// Source: DRAM bank 0 offset 0 (data content irrelevant; ISR timing is the goal).

#include "dfb_implicit_read_helper.h"

void kernel_main() {
    uint32_t dm_id;
    asm volatile("csrr %0, mhartid" : "=r"(dm_id));

    Noc noc;

    // auto issue_and_finish = [&](uint16_t id) {
    //     DataflowBuffer dfb(id);
    //     dfb_issue_implicit_read(noc, dfb);
    //     // dfb.finish();
    // };

    // if (dm_id == 4) {
    //     // 1Sx1S DFBs 0–3 (→ Neo0) and 8–11 (→ Neo2)
    //     issue_and_finish(0);  issue_and_finish(1);
    //     issue_and_finish(2);  issue_and_finish(3);
    //     issue_and_finish(8);  issue_and_finish(9);
    //     issue_and_finish(10); issue_and_finish(11);
    //     // 1Sx2A DFBs 24–27 (→ {Neo1, Neo3})
    //     issue_and_finish(24); issue_and_finish(25);
    //     issue_and_finish(26); issue_and_finish(27);
    // } else if (dm_id == 5) {
    //     // 1Sx1S DFBs 4–7 (→ Neo1) and 12–15 (→ Neo3)
    //     issue_and_finish(4);  issue_and_finish(5);
    //     issue_and_finish(6);  issue_and_finish(7);
    //     issue_and_finish(12); issue_and_finish(13);
    //     issue_and_finish(14); issue_and_finish(15);
    //     // 1Sx2A DFBs 28–31 (→ {Neo1, Neo3})
    //     issue_and_finish(28); issue_and_finish(29);
    //     issue_and_finish(30); issue_and_finish(31);
    // } else if (dm_id == 2) {
    //     // 1Sx2A DFBs 16–19 (→ {Neo0, Neo2})
    //     issue_and_finish(16); issue_and_finish(17);
    //     issue_and_finish(18); issue_and_finish(19);
    // } else if (dm_id == 3) {
    //     // 1Sx2A DFBs 20–23 (→ {Neo0, Neo2})
    //     issue_and_finish(20); issue_and_finish(21);
    //     issue_and_finish(22); issue_and_finish(23);
    // }
    // DM0/DM1 coordinators; DM6–DM7 unused.
}
