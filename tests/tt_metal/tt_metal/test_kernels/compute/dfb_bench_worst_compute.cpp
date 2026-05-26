// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for the worst-case DFB ISR latency benchmark.
//
// Acts as ALL CONSUMER on three DM→Tensix DFBs (4Sx4A each):
//   dfb::in0  - 4Sx4A DM→Tensix, ALL consumer
//   dfb::in1  - 4Sx4A DM→Tensix, ALL consumer
//   dfb::in2  - 4Sx4A DM→Tensix, ALL consumer
//
// Four Neo consumers per DFB (1-to-4 remapper fan-out) combined with three DFBs
// represents the hardware worst case: 15/16 DM-visible TCs per tensix and
// 48 set_clientR_slot writes (12 remapper entries × 4 consumers each).
//
// Threshold table (each 4Sx4A DFB, num_entries=16, num_producers=4):
//   tiles_to_post = 2 per Neo per DFB after ISR fires
//
// Each Neo thread waits for 1 credit (ISR fired), pops 2 (all posted credits)
// for each DFB, then calls finish().

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer in2(dfb::in2);

    // Wait for ISR to fire (tiles_to_post=2); pop 2 credits to unblock DM finish()
    in0.wait_front(1); in0.pop_front(1);
    in1.wait_front(1); in1.pop_front(1);
    in2.wait_front(1); in2.pop_front(1);

    // in0.finish();
    // in1.finish();
    // in2.finish();
}
