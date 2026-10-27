// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM producer kernel for BenchmarkCaseSix (worst-case-four init benchmark).
//
// Six DM threads (DM0–DM5) run this kernel. Each DM checks its mhartid and
// issues one implicit NOC read per DFB it produces, then calls
// dfb_finish_single_implicit_read_producer() (no collective thread sync).
// DFB IDs use the low-level uint16_t constructor (explicit risc masks, no bindings).
//
// Drains the full ring per DFB (num_entries=1): 1 implicit read + finish per DFB.
//
// DFB layout (24 DFBs total, 16×1Sx2A + 8×1Sx1A):
//
//   1Sx2A (DFBs 0–15): STRIDED producer, ALL consumer — 1-to-2 remapper entry each.
//     DFB  0– 3 : DM2 → {Neo0, Neo2}
//     DFB  4– 7 : DM3 → {Neo0, Neo2}
//     DFB  8–11 : DM4 → {Neo1, Neo3}
//     DFB 12–15 : DM5 → {Neo1, Neo3}
//
//   1Sx1A (DFBs 16–23): STRIDED producer, ALL consumer — 1-to-1 remapper entry each.
//     DFB 16–17 : DM2 → Neo1
//     DFB 18–19 : DM3 → Neo1
//     DFB 20–21 : DM2 → Neo3
//     DFB 22–23 : DM4 → Neo0
//
// Quasar launches this kernel on num_threads_per_cluster=6 DM harts, but each DFB
// has a single producer. Use dfb_finish_single_implicit_read_producer() instead of
// finish() to skip the collective sync_threads barrier in handle_final_credits.

#include "dfb_bench_finish_helper.h"
#include "dfb_implicit_read_helper.h"

void kernel_main() {
    uint32_t dm_id;
    asm volatile("csrr %0, mhartid" : "=r"(dm_id));

    Noc noc;

    auto issue_and_finish = [&](uint16_t id) {
        DataflowBuffer dfb(id);
        // dfb_issue_implicit_read(noc, dfb);
        // dfb_finish_single_implicit_read_producer(dfb);
    };

    if (dm_id == 4) {
        issue_and_finish(8);
        issue_and_finish(9);
        issue_and_finish(10);
        issue_and_finish(11);
        issue_and_finish(22);
        issue_and_finish(23);
    } else if (dm_id == 5) {
        issue_and_finish(12);
        issue_and_finish(13);
        issue_and_finish(14);
        issue_and_finish(15);
    } else if (dm_id == 2) {
        issue_and_finish(0);
        issue_and_finish(1);
        issue_and_finish(2);
        issue_and_finish(3);
        issue_and_finish(16);
        issue_and_finish(17);
        issue_and_finish(20);
        issue_and_finish(21);
    } else if (dm_id == 3) {
        issue_and_finish(4);
        issue_and_finish(5);
        issue_and_finish(6);
        issue_and_finish(7);
        issue_and_finish(18);
        issue_and_finish(19);
    }
    // DM0/DM1 coordinators; DM6–DM7 unused.
}
