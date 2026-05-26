// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute consumer kernel for BenchmarkWorstCaseFour ISR latency benchmark.
//
// Four Neo threads (Neo0–Neo3) run this kernel. Each Neo checks its NEO_ID CSR
// and issues wait_front(1)+pop_front(1)+finish() for the DFBs it consumes.
// DFB IDs are addressed via the low-level uint16_t constructor; no generated
// binding constants are used.
//
// Threshold table (all DFBs, num_entries=1, num_producers=1):
//   tiles_to_post = 1: each ISR posts 1 credit; wait_front(1)+pop_front(1) is correct.
//
// DFB layout (24 DFBs total, 16×1Sx2A + 8×1Sx1A):
//
//   1Sx2A (DFBs 0–15): ALL consumer — each Neo consumes its slice from both groups.
//     {Neo0, Neo2} share: DFBs 0–7   (produced by DM2 and DM3)
//     {Neo1, Neo3} share: DFBs 8–15  (produced by DM0 and DM1)
//
//   1Sx1A (DFBs 16–23): ALL consumer (single Neo per DFB, 1-to-1 remapper).
//     Neo1: DFBs 16–19  (DM2 → 16–17, DM3 → 18–19)
//     Neo3: DFBs 20–21  (DM2 → Neo3)
//     Neo0: DFBs 22–23  (DM0 → Neo0)

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();

    auto wait_pop_finish = [](uint16_t id) {
        DataflowBuffer dfb(id);
        dfb.wait_front(1);
        dfb.pop_front(1);
        // dfb.finish();
    };

    if (neo_id == 0) {
        // 1Sx2A: DFBs 0–7 shared with Neo2 (DM2 and DM3 → {Neo0, Neo2})
        wait_pop_finish(0);  wait_pop_finish(1);
        wait_pop_finish(2);  wait_pop_finish(3);
        wait_pop_finish(4);  wait_pop_finish(5);
        wait_pop_finish(6);  wait_pop_finish(7);
        // 1Sx1A: DFBs 22–23 (DM0 → Neo0, 1-to-1 remapper)
        wait_pop_finish(22); wait_pop_finish(23);
    } else if (neo_id == 1) {
        // 1Sx2A: DFBs 8–15 shared with Neo3 (DM0 and DM1 → {Neo1, Neo3})
        wait_pop_finish(8);  wait_pop_finish(9);
        wait_pop_finish(10); wait_pop_finish(11);
        wait_pop_finish(12); wait_pop_finish(13);
        wait_pop_finish(14); wait_pop_finish(15);
        // 1Sx1A: DFBs 16–17 (DM2 → Neo1) and DFBs 18–19 (DM3 → Neo1)
        wait_pop_finish(16); wait_pop_finish(17);
        wait_pop_finish(18); wait_pop_finish(19);
    } else if (neo_id == 2) {
        // 1Sx2A: DFBs 0–7 shared with Neo0 (DM2 and DM3 → {Neo0, Neo2})
        wait_pop_finish(0);  wait_pop_finish(1);
        wait_pop_finish(2);  wait_pop_finish(3);
        wait_pop_finish(4);  wait_pop_finish(5);
        wait_pop_finish(6);  wait_pop_finish(7);
        // No 1Sx1A DFBs for Neo2.
    } else {
        // neo_id == 3
        // 1Sx2A: DFBs 8–15 shared with Neo1 (DM0 and DM1 → {Neo1, Neo3})
        wait_pop_finish(8);  wait_pop_finish(9);
        wait_pop_finish(10); wait_pop_finish(11);
        wait_pop_finish(12); wait_pop_finish(13);
        wait_pop_finish(14); wait_pop_finish(15);
        // 1Sx1A: DFBs 20–21 (DM2 → Neo3)
        wait_pop_finish(20); wait_pop_finish(21);
    }
}
