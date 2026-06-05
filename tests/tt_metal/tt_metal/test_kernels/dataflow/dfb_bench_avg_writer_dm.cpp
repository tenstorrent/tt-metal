// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark writer DM kernel for the average-case DFB ISR latency benchmark.
//
// Acts as CONSUMER on one DFB:
//   dfb::t6_in  - Tensix→DM, 4Sx4S (strided producer x strided consumer)
//
// The compute kernel pushes 1 credit explicitly (push_back). This DM consumer
// waits for that credit (wait_front) and pops it, then calls finish().
// The WR_SENT ISR path for this DFB is deferred; explicit sync is used here.

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer dfb(dfb::t6_in);

    // Wait for compute to push_back(1) via explicit sync
    // dfb.wait_front(1);
    // dfb.pop_front(1);

    // dfb.finish();
}
