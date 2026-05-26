// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute consumer kernel for BenchmarkWorstCaseThree ISR latency benchmark.
//
// Four Neo threads (Neo0–Neo3) run this kernel. Each Neo checks its NEO_ID CSR
// and issues wait_front(1)+pop_front(1)+finish() for the DFBs it consumes.
// DFB IDs are addressed via the low-level uint16_t constructor; no generated
// binding constants are used.
//
// DFB layout (32 DFBs total, 16×1Sx1S + 16×1Sx2A):
//   1Sx1S (DFBs 0–15): STRIDED consumer — each Neo consumes its own slice
//     Neo0: DFBs  0– 3   (produced by DM4)
//     Neo1: DFBs  4– 7   (produced by DM5)
//     Neo2: DFBs  8–11   (produced by DM4)
//     Neo3: DFBs 12–15   (produced by DM5)
//
//   1Sx2A (DFBs 16–31): ALL consumer — two Neos share each group via remapper
//     DFBs 16–23 → Neo0 + Neo2  (produced by DM2 and DM3)
//     DFBs 24–31 → Neo1 + Neo3  (produced by DM0 and DM1)
//
// Threshold table (all DFBs, num_entries=1, num_producers=1):
//   tiles_to_post = 1: each ISR posts 1 credit; wait_front(1)+pop_front(1) is correct.

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
        // 1Sx1S: DFBs 0–3 from DM4
        wait_pop_finish(0);  wait_pop_finish(1);
        wait_pop_finish(2);  wait_pop_finish(3);
        // 1Sx2A type-X: DFBs 16–23 shared with Neo2
        wait_pop_finish(16); wait_pop_finish(17);
        wait_pop_finish(18); wait_pop_finish(19);
        wait_pop_finish(20); wait_pop_finish(21);
        wait_pop_finish(22); wait_pop_finish(23);
    } else if (neo_id == 1) {
        // 1Sx1S: DFBs 4–7 from DM5
        wait_pop_finish(4);  wait_pop_finish(5);
        wait_pop_finish(6);  wait_pop_finish(7);
        // 1Sx2A type-Y: DFBs 24–31 shared with Neo3
        wait_pop_finish(24); wait_pop_finish(25);
        wait_pop_finish(26); wait_pop_finish(27);
        wait_pop_finish(28); wait_pop_finish(29);
        wait_pop_finish(30); wait_pop_finish(31);
    } else if (neo_id == 2) {
        // 1Sx1S: DFBs 8–11 from DM4
        wait_pop_finish(8);  wait_pop_finish(9);
        wait_pop_finish(10); wait_pop_finish(11);
        // 1Sx2A type-X: DFBs 16–23 shared with Neo0
        wait_pop_finish(16); wait_pop_finish(17);
        wait_pop_finish(18); wait_pop_finish(19);
        wait_pop_finish(20); wait_pop_finish(21);
        wait_pop_finish(22); wait_pop_finish(23);
    } else {
        // neo_id == 3
        // 1Sx1S: DFBs 12–15 from DM5
        wait_pop_finish(12); wait_pop_finish(13);
        wait_pop_finish(14); wait_pop_finish(15);
        // 1Sx2A type-Y: DFBs 24–31 shared with Neo1
        wait_pop_finish(24); wait_pop_finish(25);
        wait_pop_finish(26); wait_pop_finish(27);
        wait_pop_finish(28); wait_pop_finish(29);
        wait_pop_finish(30); wait_pop_finish(31);
    }
}
