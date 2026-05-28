// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM producer kernel for BenchmarkWorstCaseFour ISR latency benchmark.
//
// Five DM threads (DM1–DM4) run this kernel. Each DM checks its mhartid and
// issues one implicit NOC read per DFB it produces, then calls finish().
// DFB IDs are addressed via the low-level uint16_t constructor since
// kernel_bindings_generated.h has no constants (DFBs are created with
// explicit risc masks, not kernel bindings).
//
// Threshold table (all DFBs, num_entries=1, num_producers=1):
//   per_txn = 1, tiles_to_post = 1: 1 read per DFB fires the ISR.
//
// DFB layout (24 DFBs total, 16×1Sx2A + 8×1Sx1A):
//
//   1Sx2A (DFBs 0–15): STRIDED producer, ALL consumer — 1-to-2 remapper entry each.
//     DFB  0– 3 : DM2 → {Neo0, Neo2}
//     DFB  4– 7 : DM3 → {Neo0, Neo2}
//     DFB  8–11 : DM4 → {Neo1, Neo3}
//     DFB 12–15 : DM1 → {Neo1, Neo3}
//
//   1Sx1A (DFBs 16–23): STRIDED producer, ALL consumer — 1-to-1 remapper entry each.
//     DFB 16–17 : DM2 → Neo1
//     DFB 18–19 : DM3 → Neo1
//     DFB 20–21 : DM2 → Neo3
//     DFB 22–23 : DM4 → Neo0
//
// TC budget (16 per tensix, 64 total):
//   t0: 4(DM4 prod 1Sx2A) + 4(Neo0 cons from DM2) + 4(Neo0 cons from DM3) + 2(DM4 prod 22-23) + 2(Neo0 cons 22-23) = 16
//   t1: 4(DM1 prod 1Sx2A) + 4(Neo1 cons from DM4) + 4(Neo1 cons from DM1) + 2(Neo1 cons 16-17) + 2(Neo1 cons 18-19) = 16
//   t2: 4(DM2 prod 1Sx2A) + 4(Neo2 cons from DM2) + 4(Neo2 cons from DM3) + 2(DM2 prod 16-17) + 2(DM2 prod 20-21) = 16
//   t3: 4(DM3 prod 1Sx2A) + 4(Neo3 cons from DM4) + 4(Neo3 cons from DM1) + 2(DM3 prod 18-19) + 2(Neo3 cons 20-21) = 16
//
// Remapper: 16 × 1-to-2 (all 16 one-to-many slots) + 8 × 1-to-1 = 24 entries, 40 set_clientR_slot writes.

#include "dfb_implicit_read_helper.h"

void kernel_main() {
    uint32_t dm_id;
    asm volatile("csrr %0, mhartid" : "=r"(dm_id));

    Noc noc;

    auto issue_and_finish = [&](uint16_t id) {
        DataflowBuffer dfb(id);
        dfb_issue_implicit_read(noc, dfb);
        // dfb.finish();
    };

    if (dm_id == 4) {
        // 1Sx2A DFBs 8–11 (→ {Neo1, Neo3})
        issue_and_finish(8);  issue_and_finish(9);
        issue_and_finish(10); issue_and_finish(11);
        // 1Sx1A DFBs 22–23 (→ Neo0)
        issue_and_finish(22); issue_and_finish(23);
    } else if (dm_id == 1) {
        // 1Sx2A DFBs 12–15 (→ {Neo1, Neo3})
        issue_and_finish(12); issue_and_finish(13);
        issue_and_finish(14); issue_and_finish(15);
    } else if (dm_id == 2) {
        // 1Sx2A DFBs 0–3 (→ {Neo0, Neo2})
        issue_and_finish(0);  issue_and_finish(1);
        issue_and_finish(2);  issue_and_finish(3);
        // 1Sx1A DFBs 16–17 (→ Neo1) and 20–21 (→ Neo3)
        issue_and_finish(16); issue_and_finish(17);
        issue_and_finish(20); issue_and_finish(21);
    } else if (dm_id == 3) {
        // 1Sx2A DFBs 4–7 (→ {Neo0, Neo2})
        issue_and_finish(4);  issue_and_finish(5);
        issue_and_finish(6);  issue_and_finish(7);
        // 1Sx1A DFBs 18–19 (→ Neo1)
        issue_and_finish(18); issue_and_finish(19);
    }
    // DM0 and DM5–DM7 are not used; they participate in no DFBs and do nothing.
}
