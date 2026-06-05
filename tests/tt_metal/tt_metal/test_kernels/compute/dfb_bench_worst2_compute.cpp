// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for the worst-case-two DFB ISR latency benchmark.
//
// Single compute kernel with num_threads=4 (Neo0,1,2,3 all run this code).
// Acts as ALL consumer on 12 DFBs: all four Neo threads receive every DFB
// via the 1-to-4 remapper fan-out (12 × 1Sx4A configuration).
//
// Threshold table (each 1Sx4A DFB, num_entries=16, num_producers=1):
//   num_txn_ids   = 2
//   hw_threshold  = 8  (16 / 2; single DM issues all 8)
//   per_txn       = 8  (8 / 1 producer)
//   tiles_to_post = 8  (8 / 1 TC per risc)
//
// With ALL pattern, ALL 4 Neo threads are consumers on ALL 12 DFBs.
// Each Neo must wait for 1 credit (ISR has fired) and pop 8 credits (all posted)
// from every DFB to allow the corresponding DM finish() to return.

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer in2(dfb::in2);
    DataflowBuffer in3(dfb::in3);
    DataflowBuffer in4(dfb::in4);
    DataflowBuffer in5(dfb::in5);
    DataflowBuffer in6(dfb::in6);
    DataflowBuffer in7(dfb::in7);
    DataflowBuffer in8(dfb::in8);
    DataflowBuffer in9(dfb::in9);
    DataflowBuffer in10(dfb::in10);
    DataflowBuffer in11(dfb::in11);

    // Each Neo waits on ALL 12 DFBs; ISR posts 8 credits per DFB per Neo.
    // Pop 8 credits per DFB to unblock the corresponding DM finish().
    // in0.wait_front(1);  in0.pop_front(1);
    // in1.wait_front(1);  in1.pop_front(1);
    // in2.wait_front(1);  in2.pop_front(1);
    // in3.wait_front(1);  in3.pop_front(1);
    // in4.wait_front(1);  in4.pop_front(1);
    // in5.wait_front(1);  in5.pop_front(1);
    // in6.wait_front(1);  in6.pop_front(1);
    // in7.wait_front(1);  in7.pop_front(1);
    // in8.wait_front(1);  in8.pop_front(1);
    // in9.wait_front(1);  in9.pop_front(1);
    // in10.wait_front(1); in10.pop_front(1);
    // in11.wait_front(1); in11.pop_front(1);

    // in0.finish();  in1.finish();  in2.finish();  in3.finish();
    // in4.finish();  in5.finish();  in6.finish();  in7.finish();
    // in8.finish();  in9.finish();  in10.finish(); in11.finish();
}
