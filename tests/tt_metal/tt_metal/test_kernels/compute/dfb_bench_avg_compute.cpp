// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark compute kernel for the average-case DFB ISR latency benchmark.
//
// Participates in three DFBs:
//   dfb::ss_in   - DM→Tensix, 4Sx4S STRIDED consumer (paired with reader_dm ss_out)
//   dfb::sa_in   - DM→Tensix, 4Sx4A ALL consumer     (paired with reader_dm sa_out)
//   dfb::t6_out  - Tensix→DM, STRIDED producer        (paired with writer_dm t6_in)
//
// Threshold table (ss_in and sa_in, num_entries=16, num_producers=4):
//   tiles_to_post = 2 per Neo per DFB after ISR fires
//
// Each Neo thread:
//   - Waits until 1 credit is available on ss_in (ISR has fired), pops 2 to drain
//   - Waits until 1 credit is available on sa_in (ISR has fired), pops 2 to drain
//   - Pushes 1 credit to t6_out to signal the writer DM (explicit sync)
//   - Calls finish() on all three DFBs

#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer ss_in(dfb::ss_in);
    DataflowBuffer sa_in(dfb::sa_in);
    DataflowBuffer t6_out(dfb::t6_out);

    // Wait for ISR to fire and post 2 credits; pop all 2 to unblock DM finish()
    ss_in.wait_front(1);
    ss_in.pop_front(1);

    sa_in.wait_front(1);
    sa_in.pop_front(1);

    // Push 1 credit to the writer DM via explicit sync
    t6_out.reserve_back(1);
    t6_out.push_back(1);

    // ss_in.finish();
    // sa_in.finish();
    // t6_out.finish();
}
